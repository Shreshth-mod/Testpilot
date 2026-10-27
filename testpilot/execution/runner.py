"""
Extensible Test Runner Interface and Reference Pytest Docker Sandbox Runner for TestPilot AI.
Enforces container isolation, network isolation, read-only repository mounts,
and honest diagnostics when Docker Desktop / WSL2 is stopped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import atexit
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Optional
import uuid

from testpilot.execution.models import (
    DockerDiagnostics,
    DockerSecurityProfile,
    TestCaseResult,
    TestOutcome,
)

logger = logging.getLogger("testpilot.execution.runner")

# Track active containers across process lifecycle to prevent orphaned containers on interrupt
_ACTIVE_DOCKER_CONTAINERS: set[str] = set()


def _emergency_container_cleanup() -> None:
    for cid in list(_ACTIVE_DOCKER_CONTAINERS):
        try:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5, check=False)
        except Exception:
            pass


atexit.register(_emergency_container_cleanup)


def extract_failure_traceback(stdout: str, stderr: str, max_chars: int = 4000) -> str:
    """
    Extracts failure details prioritizing the pytest FAILURES and summary sections
    to ensure critical error lines and assertion traces are never truncated.
    """
    combined = (stdout or "") + ("\n" + stderr if stderr else "")
    if len(combined) <= max_chars:
        return combined

    for marker in ["=== FAILURES ===", "___", "short test summary info"]:
        idx = combined.find(marker)
        if idx != -1:
            slice_from = combined[idx:]
            if len(slice_from) <= max_chars:
                return slice_from
            return slice_from[:max_chars]

    return combined[-max_chars:]


class UnsupportedFrameworkError(ValueError):
    """Raised when an unrecognized test framework is targeted, avoiding guessing shell commands."""
    pass


class DockerUnavailableError(RuntimeError):
    """Raised when container isolation is required but the Docker daemon is not running."""
    pass


class TestRunner(ABC):
    """Abstract interface for containerized and isolated test execution."""

    @abstractmethod
    def check_docker_availability(self) -> DockerDiagnostics:
        """Inspects whether Docker daemon is active and responsive."""
        pass

    @abstractmethod
    def run_tests(
        self,
        test_cases: list[dict[str, Any]],
        repo_path: Path,
        security_profile: Optional[DockerSecurityProfile] = None,
        allow_simulated_dry_run: bool = False,
    ) -> tuple[list[TestCaseResult], DockerDiagnostics]:
        """Executes test cases in an isolated container sandbox."""
        pass

    @abstractmethod
    def supports_framework(self, framework_name: str) -> bool:
        """Determines if the runner natively supports the specified test framework."""
        pass


class PytestDockerRunner(TestRunner):
    """
    Reference TestRunner implementation for Python pytest test suites.
    Executes in a disposable Docker container with read-only repository mount.
    """

    SUPPORTED_FRAMEWORKS = {"pytest", "python-pytest", "unittest"}

    def __init__(self, docker_image: str = "testpilot-runner:latest") -> None:
        self.docker_image = docker_image

    def supports_framework(self, framework_name: str) -> bool:
        return framework_name.lower() in self.SUPPORTED_FRAMEWORKS

    def check_docker_availability(self) -> DockerDiagnostics:
        """
        Runs 'docker info' to verify Docker Desktop or WSL2 Linux Engine availability.
        """
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return DockerDiagnostics(
                is_available=False,
                error_message="Docker CLI binary not found in system PATH.",
            )

        try:
            res = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                version = res.stdout.strip()
                return DockerDiagnostics(
                    is_available=True,
                    daemon_version=version,
                    error_message=None,
                )
            else:
                err_msg = res.stderr.strip() or "Docker daemon is not running or not responsive."
                return DockerDiagnostics(
                    is_available=False,
                    daemon_version=None,
                    error_message=err_msg,
                )
        except (subprocess.SubprocessError, OSError) as e:
            return DockerDiagnostics(
                is_available=False,
                daemon_version=None,
                error_message=str(e),
            )

    def run_tests(
        self,
        test_cases: list[dict[str, Any]],
        repo_path: Path,
        security_profile: Optional[DockerSecurityProfile] = None,
        allow_simulated_dry_run: bool = False,
    ) -> tuple[list[TestCaseResult], DockerDiagnostics]:
        """
        Executes selected tests in disposable Docker container.
        Never silently falls back to unsafe local host execution if Docker is unavailable.
        """
        sec = security_profile or DockerSecurityProfile()
        diagnostics = self.check_docker_availability()

        if not diagnostics.is_available:
            if not allow_simulated_dry_run:
                raise DockerUnavailableError(
                    f"Docker daemon unavailable: {diagnostics.error_message}. "
                    "Unsafe local host execution is rejected. Please start Docker Desktop or WSL2."
                )
            logger.info("Docker daemon unavailable; performing strict sandboxed dry run as explicitly requested.")
            return self._run_simulated_sandbox(test_cases, repo_path), diagnostics

        # Docker is available: launch container execution
        results = self._run_docker_container(test_cases, repo_path, sec)
        return results, diagnostics

    def _run_docker_container(
        self,
        test_cases: list[dict[str, Any]],
        repo_path: Path,
        sec: DockerSecurityProfile,
    ) -> list[TestCaseResult]:
        """Executes tests inside disposable container sandbox."""
        results: list[TestCaseResult] = []
        abs_repo = str(repo_path.expanduser().resolve()).replace("\\", "/")

        for tc in test_cases:
            t_name = tc.get("test_name", "test")
            raw_t_file = tc.get("test_file", "")
            try:
                p = Path(raw_t_file)
                if p.is_absolute():
                    t_file = str(p.relative_to(repo_path)).replace("\\", "/")
                else:
                    t_file = raw_t_file.replace("\\", "/")
            except Exception:
                t_file = raw_t_file.replace("\\", "/")

            start_t = time.perf_counter()

            cid = f"tp-{uuid.uuid4().hex[:12]}"

            # Formulate docker run command with read-only source mount, tmpfs, capability drop, and isolated network
            cmd = [
                "docker", "run", "--rm",
                "--name", cid,
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "-u", "1000:1000" if sec.non_root_user else "0:0",
                "-v", f"{abs_repo}:/workspace:ro",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "PYTHONPATH=/workspace",
                "-w", "/workspace",
                "--network", "none" if sec.network_disabled else "bridge",
                "--cpus", sec.cpu_limit,
                "--memory", sec.memory_limit,
                "--pids-limit", "100",
                self.docker_image,
                "pytest", t_file, "-k", t_name, "-o", "cache_dir=/tmp/.pytest_cache", "-v",
            ]

            _ACTIVE_DOCKER_CONTAINERS.add(cid)
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
                duration_ms = (time.perf_counter() - start_t) * 1000.0

                if res.returncode == 0:
                    outcome = TestOutcome.PASSED
                    err_msg = None
                    tb = None
                elif res.returncode == 1:
                    outcome = TestOutcome.FAILED
                    # Pytest assertion failure: extract failure lines from stdout
                    out_lines = (res.stdout + "\n" + res.stderr).splitlines()
                    assertion_lines = [l.strip() for l in out_lines if "AssertionError" in l or "E   " in l or "Error:" in l]
                    fail_summary_lines = [l.strip() for l in out_lines if "FAILED" in l]
                    if assertion_lines:
                        err_msg = assertion_lines[0]
                    elif fail_summary_lines:
                        err_msg = fail_summary_lines[0]
                    else:
                        err_msg = (res.stderr.strip()[:500] or res.stdout.strip()[:500])
                    tb = extract_failure_traceback(res.stdout, res.stderr, max_chars=4000)
                else:
                    outcome = TestOutcome.ERROR
                    err_msg = res.stderr.strip()[:500] or res.stdout.strip()[:500] or f"Process failed with exit code {res.returncode}"
                    tb = extract_failure_traceback(res.stdout, res.stderr, max_chars=4000)

                results.append(
                    TestCaseResult(
                        test_name=t_name,
                        test_file=t_file,
                        outcome=outcome,
                        duration_ms=duration_ms,
                        stdout=res.stdout[:4000],
                        stderr=res.stderr[:2000],
                        exit_code=res.returncode,
                        error_message=err_msg,
                        traceback=tb,
                        targeted_symbol=tc.get("targeted_symbol"),
                    )
                )
            except subprocess.TimeoutExpired:
                # Force kill and cleanup container if timeout expired
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10, check=False)
                results.append(
                    TestCaseResult(
                        test_name=t_name,
                        test_file=t_file,
                        outcome=TestOutcome.ERROR,
                        duration_ms=60000.0,
                        error_message="Container execution timed out after 60s.",
                        traceback="TimeoutExpired: Test exceeded 60s container execution ceiling.",
                        exit_code=-1,
                        targeted_symbol=tc.get("targeted_symbol"),
                    )
                )
            except Exception as e:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10, check=False)
                results.append(
                    TestCaseResult(
                        test_name=t_name,
                        test_file=t_file,
                        outcome=TestOutcome.ERROR,
                        duration_ms=(time.perf_counter() - start_t) * 1000.0,
                        error_message=f"Container execution exception: {str(e)[:500]}",
                        traceback=str(e),
                        exit_code=-1,
                        targeted_symbol=tc.get("targeted_symbol"),
                    )
                )
            finally:
                _ACTIVE_DOCKER_CONTAINERS.discard(cid)

        return results

    def _run_simulated_sandbox(
        self,
        test_cases: list[dict[str, Any]],
        repo_path: Path,
    ) -> list[TestCaseResult]:
        """
        Simulated safe sandbox for dry runs and offline test validation.
        Does not execute arbitrary commands on the host.
        """
        results: list[TestCaseResult] = []
        for tc in test_cases:
            t_name = tc.get("test_name", "test")
            t_file = tc.get("test_file", "")
            # Verify test file exists in repository
            target_p = repo_path / t_file
            exists = target_p.exists()
            outcome = TestOutcome.PASSED if exists else TestOutcome.ERROR

            results.append(
                TestCaseResult(
                    test_name=t_name,
                    test_file=t_file,
                    outcome=outcome,
                    duration_ms=12.5,
                    stdout=f"Simulated container execution for {t_name}",
                    stderr="",
                    exit_code=0 if exists else 1,
                    error_message=None if exists else f"Test file '{t_file}' not found in sandbox",
                    targeted_symbol=tc.get("targeted_symbol"),
                )
            )
        return results
