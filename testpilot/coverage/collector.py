"""
Runtime Coverage Collector for TestPilot Test Coverage Intelligence.
Integrates Python's coverage.py to capture empirical line-level execution,
distinguishing measured execution from static or inferred relationships.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Optional
import uuid

import coverage

from testpilot.coverage.models import (
    CoverageCollectionConfig,
    CoverageReport,
    CoverageStatus,
    LineCoverageData,
)
from testpilot.coverage.storage import CoverageStore

logger = logging.getLogger("testpilot.coverage.collector")


class CoverageCollector:
    """
    Executes Python tests under coverage.py supervision, collecting exact
    line-level execution data without altering application source code.
    """

    def __init__(
        self,
        store: Optional[CoverageStore] = None,
        config: Optional[CoverageCollectionConfig] = None,
    ) -> None:
        self.store = store or CoverageStore.get_instance()
        self.config = config or CoverageCollectionConfig()

    def collect(
        self,
        repo_path: Path | str,
        test_cases: Optional[list[dict[str, Any]]] = None,
        is_full_suite: bool = False,
        repo_id: str = "default",
        commit_hash: str = "HEAD",
        workflow_id: Optional[str] = None,
        execution_run_id: Optional[str] = None,
        execution_mode: str = "local_subprocess",
        source_dirs: Optional[list[str]] = None,
        custom_test_args: Optional[list[str]] = None,
    ) -> CoverageReport:
        """
        Executes test cases with line-level runtime coverage tracking.
        Supports both targeted test cases and full repository test suites.
        """
        abs_repo = Path(repo_path).expanduser().resolve()
        run_id = f"cov-{int(time.time())}-{uuid.uuid4().hex[:6]}"

        # Check for unsupported execution modes
        mode_lower = (execution_mode or "local_subprocess").lower()
        if mode_lower in ("simulated", "simulated_sandbox"):
            logger.warning("Simulated sandbox does not execute real code; coverage collection unsupported.")
            report = CoverageReport(
                coverage_run_id=run_id,
                workflow_id=workflow_id,
                execution_run_id=execution_run_id,
                repo_id=repo_id,
                repo_path=str(abs_repo),
                commit_hash=commit_hash,
                execution_mode="simulated_sandbox",
                is_full_suite=is_full_suite,
                status=CoverageStatus.UNSUPPORTED_MODE,
                error_message="Runtime coverage requires empirical Python execution; simulated sandbox does not run code on CPU.",
            )
            # Enrich with Phase 2 Function-Level Coverage
            try:
                from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
                report = FunctionCoverageAnalyzer.analyze_report(report, repo_path=abs_repo)
            except Exception as e:
                logger.warning(f"Could not compute function coverage analysis: {e}")

            self.store.save_report(report)
            return report

        # Resolve source directories to track
        sources = self._resolve_source_paths(abs_repo, source_dirs)

        # Create isolated temporary directory for the coverage data file
        tmp_dir = abs_repo / ".testpilot" / "tmp_coverage"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_file = tmp_dir / f"{run_id}.coverage"

        # Determine target pytest arguments
        pytest_args = self._build_pytest_args(
            abs_repo=abs_repo,
            test_cases=test_cases,
            is_full_suite=is_full_suite,
            custom_test_args=custom_test_args,
        )

        test_names = [tc.get("test_name", "test") for tc in (test_cases or [])]

        if not pytest_args:
            logger.info("No test cases or test paths specified for coverage run.")
            report = CoverageReport(
                coverage_run_id=run_id,
                workflow_id=workflow_id,
                execution_run_id=execution_run_id,
                repo_id=repo_id,
                repo_path=str(abs_repo),
                commit_hash=commit_hash,
                execution_mode=execution_mode,
                is_full_suite=is_full_suite,
                status=CoverageStatus.UNAVAILABLE,
                error_message="No test files or test cases found to execute.",
                test_cases=test_names,
            )
            # Enrich with Phase 2 Function-Level Coverage
            try:
                from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
                report = FunctionCoverageAnalyzer.analyze_report(report, repo_path=abs_repo)
            except Exception as e:
                logger.warning(f"Could not compute function coverage analysis: {e}")

            self.store.save_report(report)
            return report

        # Run pytest supervised by coverage.py
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        pp_parts = [str(abs_repo)]
        src_path = abs_repo / "src"
        if src_path.is_dir():
            pp_parts.insert(0, str(src_path))
        if "PYTHONPATH" in env and env["PYTHONPATH"]:
            pp_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pp_parts)

        python_bin = sys.executable or "python"
        cmd = [
            python_bin, "-m", "coverage", "run",
            f"--data-file={str(data_file)}",
        ]
        for s in sources:
            cmd.append(f"--source={s}")
        for o in self.config.omit_patterns:
            cmd.append(f"--omit={o}")

        cmd.extend(["-m", "pytest"])
        cmd.extend(pytest_args)

        logger.info(f"Executing coverage run: {' '.join(cmd)}")
        start_time = time.perf_counter()

        try:
            res = subprocess.run(
                cmd,
                cwd=str(abs_repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Inspect pytest returncode
            # 0: all tests passed, 1: tests failed, 2: interrupted, 3: internal error, 4: usage error, 5: no tests collected
            if res.returncode == 5:
                report = CoverageReport(
                    coverage_run_id=run_id,
                    workflow_id=workflow_id,
                    execution_run_id=execution_run_id,
                    repo_id=repo_id,
                    repo_path=str(abs_repo),
                    commit_hash=commit_hash,
                    execution_mode=execution_mode,
                    is_full_suite=is_full_suite,
                    status=CoverageStatus.UNAVAILABLE,
                    error_message=f"No tests were collected by pytest: {res.stdout.strip() or res.stderr.strip()}",
                    test_cases=test_names,
                )
                self.store.save_report(report)
                return report

            if res.returncode in (2, 3, 4):
                err_msg = res.stderr.strip() or res.stdout.strip() or f"Pytest collection or invocation error (exit {res.returncode})"
                report = CoverageReport(
                    coverage_run_id=run_id,
                    workflow_id=workflow_id,
                    execution_run_id=execution_run_id,
                    repo_id=repo_id,
                    repo_path=str(abs_repo),
                    commit_hash=commit_hash,
                    execution_mode=execution_mode,
                    is_full_suite=is_full_suite,
                    status=CoverageStatus.FAILED,
                    error_message=err_msg[:1000],
                    test_cases=test_names,
                )
                self.store.save_report(report)
                return report

            # Process collected coverage data file
            if not data_file.exists():
                report = CoverageReport(
                    coverage_run_id=run_id,
                    workflow_id=workflow_id,
                    execution_run_id=execution_run_id,
                    repo_id=repo_id,
                    repo_path=str(abs_repo),
                    commit_hash=commit_hash,
                    execution_mode=execution_mode,
                    is_full_suite=is_full_suite,
                    status=CoverageStatus.UNAVAILABLE,
                    error_message="Coverage data file was not generated by test execution.",
                    test_cases=test_names,
                )
                self.store.save_report(report)
                return report

            cov = coverage.Coverage(data_file=str(data_file))
            cov.load()

            files_map: dict[str, LineCoverageData] = {}
            total_statements = 0
            covered_statements = 0
            missing_statements = 0

            cov_data = cov.get_data()
            measured_files = cov_data.measured_files()

            for abs_file in measured_files:
                p_file = Path(abs_file)
                # Ignore test files or external files if inadvertently measured
                try:
                    rel_file = str(p_file.relative_to(abs_repo)).replace("\\", "/")
                except ValueError:
                    continue

                if self._should_omit_file(rel_file):
                    continue

                try:
                    analysis = cov.analysis2(abs_file)
                    # analysis returns (filename, executable_lines, excluded_lines, missing_lines, formatted_missing)
                    executable = sorted(list(analysis[1]))
                    excluded = sorted(list(analysis[2]))
                    missing = sorted(list(analysis[3]))
                    executed = sorted([l for l in executable if l not in missing])

                    stmt_count = len(executable)
                    cov_count = len(executed)
                    miss_count = len(missing)
                    pct = round((cov_count / max(1, stmt_count)) * 100.0, 1)

                    files_map[rel_file] = LineCoverageData(
                        file_path=rel_file,
                        absolute_path=str(p_file),
                        executed_lines=executed,
                        missing_lines=missing,
                        excluded_lines=excluded,
                        executable_lines=executable,
                        total_statements=stmt_count,
                        covered_statements=cov_count,
                        line_coverage_pct=pct,
                    )

                    total_statements += stmt_count
                    covered_statements += cov_count
                    missing_statements += miss_count
                except Exception as e:
                    logger.warning(f"Error analyzing coverage for {rel_file}: {e}")

            # Also discover and include untouched source files in the repository
            # so uncovered files are accurately reflected in the report!
            self._include_untouched_source_files(
                abs_repo=abs_repo,
                sources=sources,
                files_map=files_map,
                cov=cov,
            )

            # Recompute total aggregation including untouched files
            total_statements = sum(f.total_statements for f in files_map.values())
            covered_statements = sum(f.covered_statements for f in files_map.values())
            missing_statements = sum(len(f.missing_lines) for f in files_map.values())
            overall_pct = round((covered_statements / max(1, total_statements)) * 100.0, 1)

            report = CoverageReport(
                coverage_run_id=run_id,
                workflow_id=workflow_id,
                execution_run_id=execution_run_id,
                repo_id=repo_id,
                repo_path=str(abs_repo),
                commit_hash=commit_hash,
                execution_mode=execution_mode,
                is_full_suite=is_full_suite,
                status=CoverageStatus.SUCCESS,
                error_message=None,
                test_cases=test_names,
                total_statements=total_statements,
                covered_statements=covered_statements,
                missing_statements=missing_statements,
                line_coverage_pct=overall_pct,
                files=files_map,
                metadata={
                    "duration_ms": duration_ms,
                    "pytest_exit_code": res.returncode,
                    "measured_files_count": len(files_map),
                },
            )

            # Enrich with Phase 2 Function-Level Coverage
            try:
                from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
                report = FunctionCoverageAnalyzer.analyze_report(report, repo_path=abs_repo)
            except Exception as e:
                logger.warning(f"Could not compute function coverage analysis: {e}")

            self.store.save_report(report)
            return report

        except subprocess.TimeoutExpired:
            report = CoverageReport(
                coverage_run_id=run_id,
                workflow_id=workflow_id,
                execution_run_id=execution_run_id,
                repo_id=repo_id,
                repo_path=str(abs_repo),
                commit_hash=commit_hash,
                execution_mode=execution_mode,
                is_full_suite=is_full_suite,
                status=CoverageStatus.FAILED,
                error_message=f"Coverage test execution timed out after {self.config.timeout_seconds} seconds.",
                test_cases=test_names,
            )
            # Enrich with Phase 2 Function-Level Coverage
            try:
                from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
                report = FunctionCoverageAnalyzer.analyze_report(report, repo_path=abs_repo)
            except Exception as e:
                logger.warning(f"Could not compute function coverage analysis: {e}")

            self.store.save_report(report)
            return report
        except Exception as e:
            report = CoverageReport(
                coverage_run_id=run_id,
                workflow_id=workflow_id,
                execution_run_id=execution_run_id,
                repo_id=repo_id,
                repo_path=str(abs_repo),
                commit_hash=commit_hash,
                execution_mode=execution_mode,
                is_full_suite=is_full_suite,
                status=CoverageStatus.FAILED,
                error_message=f"Coverage collection unexpected error: {str(e)}",
                test_cases=test_names,
            )
            # Enrich with Phase 2 Function-Level Coverage
            try:
                from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
                report = FunctionCoverageAnalyzer.analyze_report(report, repo_path=abs_repo)
            except Exception as e:
                logger.warning(f"Could not compute function coverage analysis: {e}")

            self.store.save_report(report)
            return report
        finally:
            # Clean up transient .coverage data file
            if data_file.exists():
                try:
                    data_file.unlink()
                except Exception:
                    pass

    def _resolve_source_paths(self, abs_repo: Path, source_dirs: Optional[list[str]]) -> list[str]:
        """Resolves target source directories to measure."""
        if source_dirs:
            resolved = []
            for s in source_dirs:
                sp = abs_repo / s if not Path(s).is_absolute() else Path(s)
                if sp.exists():
                    resolved.append(str(sp))
            if resolved:
                return resolved

        # Default detection: check for src/ or package directories
        src_dir = abs_repo / "src"
        if src_dir.is_dir():
            return [str(src_dir)]

        # Check for top-level python packages (dirs with __init__.py excluding tests)
        packages = []
        for child in abs_repo.iterdir():
            if child.is_dir() and not child.name.startswith(".") and child.name not in ("tests", "test", "testing", "venv", ".venv", "build", "dist"):
                if (child / "__init__.py").exists():
                    packages.append(str(child))
        if packages:
            return packages

        return [str(abs_repo)]

    def _should_omit_file(self, rel_path: str) -> bool:
        """Determines if a relative path should be omitted from application coverage."""
        rel_norm = rel_path.replace("\\", "/").lower()
        if rel_norm.startswith("tests/") or "/tests/" in rel_norm or rel_norm.startswith("test/"):
            return True
        if rel_norm.startswith(".venv/") or "/.venv/" in rel_norm:
            return True
        if rel_norm.startswith("venv/") or "/venv/" in rel_norm:
            return True
        if rel_norm.endswith("conftest.py") or rel_norm.endswith("setup.py"):
            return True
        if "/test_" in rel_norm or rel_norm.startswith("test_") or rel_norm.endswith("_test.py"):
            return True
        return False

    def _build_pytest_args(
        self,
        abs_repo: Path,
        test_cases: Optional[list[dict[str, Any]]],
        is_full_suite: bool,
        custom_test_args: Optional[list[str]] = None,
    ) -> list[str]:
        """Builds exact arguments for pytest execution."""
        if custom_test_args:
            return list(custom_test_args)

        if is_full_suite or not test_cases:
            # Look for tests/ directory or test files
            tests_dir = abs_repo / "tests"
            if tests_dir.is_dir():
                return ["tests", "-v"]
            test_dir = abs_repo / "test"
            if test_dir.is_dir():
                return ["test", "-v"]
            return ["-v"]

        # Targeted test cases
        args = ["-v"]
        test_files = set()
        test_filters = []

        for tc in test_cases:
            raw_file = tc.get("test_file") or ""
            t_name = tc.get("test_name") or ""

            if raw_file:
                p = Path(raw_file)
                try:
                    if p.is_absolute():
                        rel = str(p.relative_to(abs_repo)).replace("\\", "/")
                    else:
                        rel = raw_file.replace("\\", "/")
                except Exception:
                    rel = raw_file.replace("\\", "/")
                test_files.add(rel)

            if t_name:
                test_filters.append(t_name)

        if test_files:
            args.extend(sorted(list(test_files)))
        if test_filters:
            # Combine filters with 'or'
            filter_expr = " or ".join(test_filters)
            args.extend(["-k", filter_expr])

        return args

    def _include_untouched_source_files(
        self,
        abs_repo: Path,
        sources: list[str],
        files_map: dict[str, LineCoverageData],
        cov: coverage.Coverage,
    ) -> None:
        """
        Discovers unexecuted application Python files and records them with 0% coverage.
        Ensures untouched modules are never reported as having 100% or phantom coverage!
        """
        for s in sources:
            src_p = Path(s)
            if not src_p.exists():
                continue

            py_files = src_p.rglob("*.py") if src_p.is_dir() else [src_p]
            for pf in py_files:
                try:
                    rel_p = str(pf.relative_to(abs_repo)).replace("\\", "/")
                except ValueError:
                    continue

                if self._should_omit_file(rel_p) or rel_p in files_map:
                    continue

                # Analyze lines via coverage AST analysis
                try:
                    analysis = cov.analysis2(str(pf))
                    executable = sorted(list(analysis[1]))
                    excluded = sorted(list(analysis[2]))
                    stmt_count = len(executable)

                    files_map[rel_p] = LineCoverageData(
                        file_path=rel_p,
                        absolute_path=str(pf),
                        executed_lines=[],
                        missing_lines=executable,
                        excluded_lines=excluded,
                        executable_lines=executable,
                        total_statements=stmt_count,
                        covered_statements=0,
                        line_coverage_pct=0.0,
                    )
                except Exception:
                    pass
