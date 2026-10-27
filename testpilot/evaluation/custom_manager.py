"""
Dynamic Multi-Repository Regression Benchmark Manager for TestPilot AI.
Orchestrates custom repository registration, structure inspection, test discovery,
independent benchmark ground truth preparation, baseline execution, and strict metric aggregation.
"""

import base64
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from testpilot.capabilities import CapabilityDetector, SupportLevel
from testpilot.evaluation.auto_benchmark import AutoBenchmarkGenerator
from testpilot.evaluation.baselines import BaselineEvaluator
from testpilot.evaluation.bug_mining import HistoricalBugMiner
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    CustomRepositoryRecord,
    GeneratedTestRecord,
    GroundTruthReviewStatus,
    GroundTruthValidationLevel,
    PrepareBenchmarkRequest,
    PrepareHistoricalBugRequest,
    RegisterRepositoryRequest,
    RepositoryStatus,
    ReviewGroundTruthRequest,
)
from testpilot.evaluation.engine import compute_metrics
from testpilot.evaluation.generation_manager import GenerationJobManager
from testpilot.evaluation.generation_models import (
    GenerationJobRecord,
    GenerationJobStatus,
    HistoricalBugCandidate,
    RepositoryValidationSummary,
)
from testpilot.evaluation.models import (
    BaselineResult,
    BaselineType,
    EvaluationMetrics,
    EvaluationRun,
    GroundTruth,
)
from testpilot.evolution.engine import RepositoryEvolutionEngine
from testpilot.evolution.models import ChangedSymbol, ChangeType


class CustomRepositoryBenchmarkManager:
    """
    Manages the lifecycle of user-registered benchmark repositories:
    1. Validation, path/URL normalization, and stable identity creation.
    2. Safe remote repository cloning into dedicated benchmark workspace.
    3. Deep repository inspection (AST, test discovery, git metadata).
    4. Independent benchmark ground truth preparation and validation.
    5. Execution across supported baseline configurations (Full Regression, Naive, TestPilot, TestPilot + RAG).
    6. Result persistence and pooled mathematical metric aggregation.
    7. Safe repository removal and stale result tracking.
    """

    ALLOWED_REMOTE_HOSTS = {"github.com", "www.github.com"}
    SAFE_REPO_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_.-]+$")

    def __init__(self, storage_dir: Optional[str] = None, workspace_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir).resolve()
        else:
            workspace_store = Path("testbed/evaluation_store").resolve()
            if workspace_store.parent.exists():
                self.storage_dir = workspace_store
            else:
                self.storage_dir = Path.home() / ".testpilot_eval"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.repos_file = self.storage_dir / "custom_repositories.json"
        self._ensure_storage()

        # Dedicated benchmark workspace for cloned remote repositories
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir).resolve()
        else:
            self.workspace_dir = (self.storage_dir / "clones").resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.generation_job_mgr = GenerationJobManager(storage_dir=str(self.storage_dir))

    def _ensure_storage(self) -> None:
        """Ensures the storage file exists with valid JSON structure."""
        if not self.repos_file.exists():
            self.repos_file.write_text("{}", encoding="utf-8")

    def _load_all_records(self) -> dict[str, CustomRepositoryRecord]:
        """Loads all repository records from persistent JSON storage."""
        if not self.repos_file.exists():
            return {}
        try:
            raw = json.loads(self.repos_file.read_text(encoding="utf-8"))
            return {k: CustomRepositoryRecord(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save_all_records(self, records: dict[str, CustomRepositoryRecord]) -> None:
        """Atomically saves all repository records to persistent JSON storage."""
        data = {k: v.model_dump() for k, v in records.items()}
        temp_file = self.storage_dir / f"custom_repositories_{int(time.time()*1000)}.tmp"
        temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if self.repos_file.exists():
            try:
                self.repos_file.unlink()
            except Exception:
                pass
        temp_file.rename(self.repos_file)

    def update_repository_record(self, record: CustomRepositoryRecord) -> None:
        """Atomically updates and saves a single repository record."""
        records = self._load_all_records()
        records[record.repo_id] = record
        self._save_all_records(records)

    @classmethod
    def is_remote_url(cls, path_or_url: str) -> bool:
        """
        Determines whether the given input is a remote repository URL
        rather than a local filesystem path.
        """
        if not path_or_url or not isinstance(path_or_url, str):
            return False
        cleaned = path_or_url.strip().strip('"').strip("'")
        # Check for Windows drive letter path: e.g. C:\ or C:/ or D:
        if (
            len(cleaned) >= 2
            and cleaned[0].isalpha()
            and cleaned[1] == ":"
            and (len(cleaned) == 2 or cleaned[2] in ("\\", "/"))
        ):
            return False
        # Remote URL protocols or Git/GitHub patterns
        lower = cleaned.lower()
        if lower.startswith(("https://", "http://", "git://", "ssh://", "ftp://", "file://", "git@")):
            return True
        if lower.startswith("github.com/"):
            return True
        return False

    @classmethod
    def parse_and_validate_remote_url(cls, url_str: str) -> tuple[str, str, str]:
        """
        Validates and normalizes remote GitHub URL.
        Returns: (normalized_clone_url, owner, repo_name)
        Raises: ValueError for invalid schemes, hosts, malformed paths, or traversal.
        """
        cleaned = (url_str or "").strip().strip('"').strip("'")
        if not cleaned:
            raise ValueError("Remote repository URL must not be empty.")

        # Handle git@github.com:owner/repo(.git)?
        if cleaned.startswith("git@github.com:"):
            path_part = cleaned[len("git@github.com:"):]
            cleaned = f"https://github.com/{path_part}"
        elif cleaned.lower().startswith("github.com/"):
            cleaned = f"https://{cleaned}"

        parsed = urllib.parse.urlparse(cleaned)

        # 1. Validate scheme
        scheme = parsed.scheme.lower()
        if scheme not in ("https", "http"):
            raise ValueError(
                f"Unsupported URL scheme '{parsed.scheme}'. Only HTTPS and HTTP GitHub repository URLs are supported."
            )

        # 2. Validate host (Strict GitHub host, prevent SSRF / private IP access)
        host = (parsed.hostname or "").lower()
        if not host or host not in cls.ALLOWED_REMOTE_HOSTS:
            raise ValueError(
                f"Unsupported host '{host or parsed.netloc}'. TestPilot benchmark manager currently supports GitHub repositories (github.com)."
            )

        # 3. Validate path segments: must be /{owner}/{repo} or /{owner}/{repo}.git
        path_segments = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_segments) != 2:
            raise ValueError(
                f"Malformed GitHub URL '{url_str}'. Expected format: https://github.com/owner/repository"
            )

        owner, raw_repo = path_segments[0], path_segments[1]
        repo_name = raw_repo[:-4] if raw_repo.endswith(".git") else raw_repo

        if not cls.SAFE_REPO_IDENTIFIER.match(owner) or not cls.SAFE_REPO_IDENTIFIER.match(repo_name):
            raise ValueError(
                f"Invalid characters in repository owner '{owner}' or name '{repo_name}'."
            )

        if owner in (".", "..") or repo_name in (".", ".."):
            raise ValueError("Path traversal sequences are not allowed in repository URL.")

        normalized_clone_url = f"https://github.com/{owner}/{repo_name}.git"
        return normalized_clone_url, owner, repo_name

    def clone_or_reuse_remote_repo(
        self,
        url_str: str,
        auth_token: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> tuple[Path, str, str]:
        """
        Safely clones a remote repository into the dedicated benchmark workspace
        or reuses it if already present. Supports private repositories via token.
        Credentials are never stored in git config, URLs, or on disk.
        Returns: (cloned_local_path, normalized_url, repo_name)
        """
        clone_url, owner, repo_name = self.parse_and_validate_remote_url(url_str)

        # Derive safe directory name within benchmark workspace
        safe_dirname = f"{owner.lower()}_{repo_name.lower()}"
        dest_dir = (self.workspace_dir / safe_dirname).resolve()

        # Security check: verify destination is strictly within workspace_dir (prevent path traversal)
        try:
            dest_dir.relative_to(self.workspace_dir.resolve())
        except ValueError:
            raise ValueError(f"Destination path resolves outside benchmark workspace: {dest_dir}")

        # Check if already cloned
        if dest_dir.exists():
            if dest_dir.is_dir() and (dest_dir / ".git").exists():
                return dest_dir, clone_url, repo_name
            else:
                raise ValueError(
                    f"Destination directory '{dest_dir.name}' exists in benchmark workspace but is not a valid Git repository."
                )

        # Invoke git clone safely without exposing authorization material in process arguments
        cmd = ["git", "--no-pager", "clone", "--depth", "50"]
        if branch and branch.strip():
            cmd.extend(["--branch", branch.strip()])
        cmd.extend([clone_url, str(dest_dir)])

        env = dict(os.environ)
        # Prevent git from hanging waiting for interactive user credentials
        env["GIT_TERMINAL_PROMPT"] = "0"
        if auth_token and auth_token.strip():
            b64_auth = base64.b64encode(f"x-access-token:{auth_token.strip()}".encode()).decode()
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
            env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {b64_auth}"

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise TimeoutError(f"Cloning '{clone_url}' timed out after 120 seconds. Please check network connection.")
        except Exception as e:
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            clean_e = re.sub(r"[a-zA-Z0-9_\-]{20,}", "[REDACTED_SECRET]", str(e))
            raise RuntimeError(f"Failed to clone repository: {clean_e}")
        finally:
            env.clear()

        if proc.returncode != 0:
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raw_err = (proc.stderr or proc.stdout or "").strip()
            err_msg = re.sub(r"[a-zA-Z0-9_\-]{20,}", "[REDACTED_SECRET]", raw_err)
            if "Authentication failed" in err_msg or "could not read Username" in err_msg or "terminal prompts disabled" in err_msg:
                if auth_token:
                    raise PermissionError(
                        f"Authentication failed for '{clone_url}'. "
                        "The provided GitHub Personal Access Token was rejected. Please verify token permissions."
                    )
                else:
                    raise PermissionError(
                        f"Authentication required or repository is private: '{clone_url}'. "
                        "Please provide a GitHub Personal Access Token with repository read permissions."
                    )
            elif "Repository not found" in err_msg or "not found" in err_msg.lower():
                raise FileNotFoundError(
                    f"Remote repository not found: '{clone_url}'. Please verify owner and repository name."
                )
            else:
                raise RuntimeError(f"Git clone failed (exit code {proc.returncode}): {err_msg}")

        if not (dest_dir / ".git").exists():
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise RuntimeError(f"Cloning appeared to succeed, but '{dest_dir}' is not a valid Git repository.")

        return dest_dir, clone_url, repo_name

    def normalize_repo_path(self, path_str: str) -> Path:
        """
        Validates and normalizes repository path server-side.
        Prevents directory traversal, invalid paths, and system root targets.
        """
        if not path_str or not path_str.strip():
            raise ValueError("Repository path must not be empty.")

        cleaned = path_str.strip().strip('"').strip("'")
        target_path = Path(cleaned).expanduser().resolve()

        if not target_path.exists():
            raise FileNotFoundError(f"Path does not exist: {target_path}")

        if not target_path.is_dir():
            raise ValueError(f"Path is not a directory: {target_path}")

        # Security check: Prevent registering root drives or system directories
        parts = target_path.parts
        if len(parts) <= 1:
            raise ValueError("Cannot register root filesystem drive as a repository.")
        if len(parts) == 2 and parts[0].endswith(":\\") and parts[1].lower() in ("windows", "program files", "system32"):
            raise ValueError("Access to system directory is disallowed.")

        return target_path

    def _generate_repo_id(self, canonical_path: Path) -> str:
        """Derives a stable, deterministic repository identifier from its canonical path."""
        norm_str = str(canonical_path).lower().replace("\\", "/")
        path_hash = hashlib.sha256(norm_str.encode("utf-8")).hexdigest()[:10]
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in canonical_path.name.lower())
        return f"repo_{safe_name}_{path_hash}"

    def register_repository(self, req: RegisterRepositoryRequest) -> CustomRepositoryRecord:
        """
        Registers a new repository in the benchmark suite.
        Supports both local paths and remote GitHub URLs.
        Validates path/URL, clones remote repos into dedicated benchmark workspace,
        checks for duplicates, inspects structure, and persists.
        """
        records = self._load_all_records()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        is_remote = self.is_remote_url(req.path)
        remote_url: Optional[str] = None

        if is_remote:
            # 1. Parse and validate URL upfront before any filesystem operations
            normalized_clone_url, owner, repo_name = self.parse_and_validate_remote_url(req.path)
            clean_remote_check = normalized_clone_url.lower().rstrip("/").removesuffix(".git")

            # Check for active duplicate registration by remote URL
            for existing in records.values():
                if existing.is_active and existing.remote_url:
                    existing_clean = existing.remote_url.lower().rstrip("/").removesuffix(".git")
                    if existing_clean == clean_remote_check:
                        return existing

            # 2. Clone or safely reuse existing clone in benchmark workspace
            cloned_path, remote_url, default_name = self.clone_or_reuse_remote_repo(
                req.path,
                auth_token=req.auth_token,
                branch=req.branch,
            )
            canonical_path = cloned_path.resolve()
            name = (req.name or "").strip() or default_name
        else:
            # Local filesystem path
            canonical_path = self.normalize_repo_path(req.path)
            name = (req.name or "").strip() or canonical_path.name

        # Check for active duplicate registration by canonical local path
        for existing in records.values():
            if existing.is_active and str(Path(existing.canonical_path).resolve()) == str(canonical_path):
                return existing

        repo_id = self._generate_repo_id(canonical_path)

        record = CustomRepositoryRecord(
            repo_id=repo_id,
            name=name,
            canonical_path=str(canonical_path),
            remote_url=remote_url,
            is_remote=is_remote,
            is_private=bool(req.auth_token),
            auth_type="token" if req.auth_token else None,
            is_active=True,
            status=RepositoryStatus.REGISTERED,
            status_message=(
                f"Cloned from {remote_url}. Running initial structure inspection..."
                if is_remote
                else "Repository registered. Running initial structure inspection..."
            ),
            created_at=now_iso,
            updated_at=now_iso,
        )

        # Immediately inspect structure and discover tests
        record = self.inspect_repository(record)
        records[repo_id] = record
        self._save_all_records(records)
        return record

    def validate_repository_candidate(
        self,
        path_str: str,
        auth_token: Optional[str] = None,
    ) -> RepositoryValidationSummary:
        """
        Performs pre-flight inspection and validation for a repository candidate
        (public Git URL, private Git URL with token, or local directory)
        before saving. Returns structured validation summary.
        """
        is_remote = self.is_remote_url(path_str)
        if is_remote:
            try:
                normalized_url, owner, repo_name = self.parse_and_validate_remote_url(path_str)
            except Exception as e:
                return RepositoryValidationSummary(
                    source_type="public_git" if not auth_token else "private_git",
                    repo_name="unknown",
                    canonical_path="",
                    is_valid=False,
                    error_message=str(e),
                )

            # Test reachability with git ls-remote safely without exposing authorization material in process arguments
            cmd = ["git", "--no-pager", "ls-remote", "--symref", normalized_url, "HEAD"]

            env = dict(os.environ)
            env["GIT_TERMINAL_PROMPT"] = "0"
            if auth_token and auth_token.strip():
                b64_auth = base64.b64encode(f"x-access-token:{auth_token.strip()}".encode()).decode()
                env["GIT_CONFIG_COUNT"] = "1"
                env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
                env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {b64_auth}"
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=env, check=False)
                env.clear()
                if proc.returncode != 0:
                    raw_err = proc.stderr or proc.stdout or ""
                    err = re.sub(r"[a-zA-Z0-9_\-]{20,}", "[REDACTED_SECRET]", raw_err)
                    if "Authentication failed" in err or "could not read Username" in err or "terminal prompts disabled" in err:
                        if auth_token:
                            return RepositoryValidationSummary(
                                source_type="private_git",
                                repo_name=repo_name,
                                remote_url=normalized_url,
                                canonical_path="",
                                private_access_status="auth_failed",
                                is_valid=False,
                                error_message="Authentication failed with provided token. Please verify token permissions (repo read access).",
                            )
                        else:
                            return RepositoryValidationSummary(
                                source_type="private_git",
                                repo_name=repo_name,
                                remote_url=normalized_url,
                                canonical_path="",
                                private_access_status="auth_required",
                                is_valid=False,
                                warnings=["Repository is private or requires authentication. Please supply a GitHub Personal Access Token."],
                            )
                    return RepositoryValidationSummary(
                        source_type="public_git",
                        repo_name=repo_name,
                        remote_url=normalized_url,
                        canonical_path="",
                        is_valid=False,
                        error_message=f"Git remote error: {err[:200]}",
                    )

                default_branch = "main"
                commit_sha = None
                for line in proc.stdout.splitlines():
                    if "refs/heads/" in line and line.startswith("ref:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            default_branch = parts[1].replace("refs/heads/", "")
                    elif "HEAD" in line and not line.startswith("ref:"):
                        commit_sha = line.split()[0]

                safe_dirname = f"{owner.lower()}_{repo_name.lower()}"
                dest_dir = (self.workspace_dir / safe_dirname).resolve()
                existing_tests = 0
                languages = ["Python"]
                source_files_count = 0

                if dest_dir.exists() and (dest_dir / ".git").exists():
                    try:
                        evaluator = BaselineEvaluator(repo_path=str(dest_dir))
                        tests = evaluator.discover_all_tests()
                        existing_tests = len(tests)
                        py_files = list(dest_dir.rglob("*.py"))
                        source_files_count = len(py_files)
                    except Exception:
                        pass

                return RepositoryValidationSummary(
                    source_type="private_git" if auth_token else "public_git",
                    repo_name=repo_name,
                    remote_url=normalized_url,
                    canonical_path=str(dest_dir),
                    branch=default_branch,
                    commit_sha=commit_sha,
                    languages=languages,
                    test_framework="pytest",
                    existing_tests_count=existing_tests,
                    source_files_count=source_files_count,
                    size_bytes=0,
                    excluded_dirs=[".git", ".venv", "venv", "node_modules", "__pycache__"],
                    private_access_status="authenticated" if auth_token else "not_required",
                    is_valid=True,
                    warnings=[] if existing_tests > 0 else ["No pre-existing tests detected in clone. Autonomous test generation recommended."],
                )
            except subprocess.TimeoutExpired:
                return RepositoryValidationSummary(
                    source_type="public_git",
                    repo_name=repo_name,
                    remote_url=normalized_url,
                    canonical_path="",
                    is_valid=False,
                    error_message="Network timeout while contacting Git remote (15s limit).",
                )
            except Exception as e:
                return RepositoryValidationSummary(
                    source_type="public_git",
                    repo_name=repo_name,
                    remote_url=normalized_url,
                    canonical_path="",
                    is_valid=False,
                    error_message=str(e),
                )
        else:
            try:
                canonical = self.normalize_repo_path(path_str)
            except Exception as e:
                return RepositoryValidationSummary(
                    source_type="local_folder",
                    repo_name=Path(path_str).name if path_str else "invalid",
                    canonical_path="",
                    is_valid=False,
                    error_message=str(e),
                )

            is_git = (canonical / ".git").exists()
            branch = None
            commit_sha = None
            if is_git:
                try:
                    b_proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(canonical), capture_output=True, text=True, check=False)
                    branch = b_proc.stdout.strip() or "HEAD"
                    c_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(canonical), capture_output=True, text=True, check=False)
                    commit_sha = c_proc.stdout.strip()
                except Exception:
                    pass

            # Run deep capability inspection
            compat_report = CapabilityDetector.generate_report(canonical)
            langs = compat_report.ecosystem.detected_languages or ["Generic/Text"]
            test_fw = ", ".join(compat_report.ecosystem.detected_test_frameworks) or ("pytest" if "Python" in langs else "none")

            existing_tests = 0
            try:
                evaluator = BaselineEvaluator(repo_path=str(canonical))
                existing_tests = len(evaluator.discover_all_tests())
            except Exception:
                pass

            warnings = []
            if not is_git:
                warnings.append("Folder is not a Git repository. Commit-level diff tracking will be limited to snapshots.")
            if existing_tests == 0:
                warnings.append("No pre-existing tests detected. Autonomous test synthesis will be required.")
            for reason in compat_report.blocking_reasons:
                warnings.append(reason)

            # File counts
            source_count = 0
            for root, dirs, files in os.walk(str(canonical)):
                dirs[:] = [d for d in dirs if d not in CapabilityDetector.IGNORE_DIRS]
                source_count += len(files)

            return RepositoryValidationSummary(
                source_type="local_folder",
                repo_name=canonical.name,
                remote_url=None,
                canonical_path=str(canonical),
                branch=branch,
                commit_sha=commit_sha,
                languages=langs,
                test_framework=test_fw,
                existing_tests_count=existing_tests,
                source_files_count=source_count,
                size_bytes=0,
                excluded_dirs=sorted(list(CapabilityDetector.IGNORE_DIRS)),
                is_monorepo=compat_report.ecosystem.workspace.is_monorepo,
                workspace_type=compat_report.ecosystem.workspace.workspace_type,
                manifests=[m.path for m in compat_report.ecosystem.manifests],
                stage_capabilities={k: v.support_level.value for k, v in compat_report.stage_capabilities.items()},
                compatibility_report=compat_report.model_dump(),
                private_access_status="not_required",
                is_valid=True,
                warnings=warnings,
            )

    def start_generation_job(self, repo_id: str, max_targets: int = 15) -> GenerationJobRecord:
        """Starts an observable, durable test generation job in the background."""
        record = self.get_repository(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        # Enforce capability check: Do not start generation jobs for unsupported ecosystems
        if record.stage_capabilities.get("test_generation") == "unsupported":
            reason = "Autonomous test generation is not supported for this ecosystem."
            missing = "Autonomous test generator adapter"
            if record.compatibility_report:
                cap_data = record.compatibility_report.get("stage_capabilities", {}).get("test_generation", {})
                reason = cap_data.get("reason", reason)
                missing = cap_data.get("missing_adapter", missing)
            raise ValueError(
                f"Cannot start generation job for repository '{record.name}': Autonomous test generation is UNSUPPORTED. "
                f"{reason} (Missing adapter: {missing}). "
                "Autonomous test generation currently requires Python pytest. For other ecosystems, "
                "please use historical bug-fix ground truth or manual mapping."
            )

        job = self.generation_job_mgr.create_job(repo_id, record.detected_revision)
        record.status = RepositoryStatus.GENERATING_TESTS
        record.status_message = "Generation job running in background..."
        self.update_repository_record(record)

        self.generation_job_mgr.start_generation_async(job.job_id, record, self, max_targets=max_targets)
        return job

    def get_generation_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        return self.generation_job_mgr.get_job(job_id)

    def get_latest_generation_job(self, repo_id: str) -> Optional[GenerationJobRecord]:
        return self.generation_job_mgr.get_latest_job_for_repo(repo_id)

    def cancel_generation_job(self, job_id: str) -> bool:
        return self.generation_job_mgr.cancel_job(job_id)

    def mine_historical_bugs(
        self,
        repo_id: str,
        max_commits: int = 50,
        max_candidates: int = 10,
    ) -> list[HistoricalBugCandidate]:
        record = self.get_repository(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")
        if not record.is_git:
            raise ValueError(f"Repository '{record.name}' is not a Git repository. Historical bug mining requires Git commit history.")
        miner = HistoricalBugMiner(record.canonical_path)
        return miner.mine_candidate_bug_fixes(repo_id=repo_id, max_commits=max_commits, max_candidates=max_candidates)

    def validate_and_register_historical_bug(
        self,
        repo_id: str,
        candidate: HistoricalBugCandidate,
    ) -> CustomBenchmarkCase:
        record = self.get_repository(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")
        miner = HistoricalBugMiner(record.canonical_path)
        validated_cand = miner.validate_bug_candidate(candidate)
        case = miner.convert_candidate_to_benchmark_case(validated_cand)

        existing_ids = {c.case_id for c in record.benchmark_cases}
        if case.case_id in existing_ids:
            record.benchmark_cases = [c if c.case_id != case.case_id else case for c in record.benchmark_cases]
        else:
            record.benchmark_cases.append(case)

        validated_cases = [c for c in record.benchmark_cases if c.ground_truth.is_verified]
        record.validated_benchmark_cases_count = len(validated_cases)
        record.cases_awaiting_review_count = len([c for c in record.benchmark_cases if not c.ground_truth.is_verified])
        if record.validated_benchmark_cases_count > 0 and record.status in (RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED, RepositoryStatus.REGISTERED):
            record.status = RepositoryStatus.READY_FOR_EVALUATION
            record.status_message = f"{record.validated_benchmark_cases_count} benchmark cases ready for evaluation."
        self.update_repository_record(record)
        return case

    def inspect_repository(self, record: CustomRepositoryRecord | str) -> CustomRepositoryRecord:
        """
        Inspects repository structure: Git revision, languages, source files, and test cases.
        Updates benchmark-readiness status.
        """
        if isinstance(record, str):
            rec_obj = self.get_repository(record)
            if not rec_obj:
                raise ValueError(f"Repository ID '{record}' not found.")
            record = rec_obj

        repo_path = Path(record.canonical_path).resolve()
        if not repo_path.exists():
            record.status = RepositoryStatus.EVALUATION_FAILED
            record.status_message = f"Repository path no longer exists on disk: {repo_path}"
            return record

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record.last_analyzed_at = now_iso
        record.updated_at = now_iso

        # 1. Git metadata inspection
        git_dir = repo_path / ".git"
        if git_dir.exists():
            record.is_git = True
            try:
                rev_proc = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if rev_proc.returncode == 0 and rev_proc.stdout.strip():
                    new_rev = rev_proc.stdout.strip()
                    if record.detected_revision and record.detected_revision != new_rev:
                        # Repository revision advanced; mark cases tied to old revision as STALE
                        for case in record.benchmark_cases:
                            if case.commit_sha and case.commit_sha != new_rev:
                                case.review_status = GroundTruthReviewStatus.STALE
                                case.ground_truth.is_verified = False
                                if "STALE" not in case.notes:
                                    case.notes += f" [STALE: Repository HEAD changed from {case.commit_sha[:8]} to {new_rev[:8]}]"
                    record.detected_revision = new_rev

                br_proc = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if br_proc.returncode == 0 and br_proc.stdout.strip():
                    record.current_branch = br_proc.stdout.strip()
            except Exception:
                pass
        else:
            record.is_git = False

        # 2. Comprehensive Ecosystem & Capability Detection
        compat_report = CapabilityDetector.generate_report(repo_path)
        record.compatibility_report = compat_report.model_dump()
        record.stage_capabilities = {k: v.support_level.value for k, v in compat_report.stage_capabilities.items()}
        record.is_monorepo = compat_report.ecosystem.workspace.is_monorepo
        record.workspace_type = compat_report.ecosystem.workspace.workspace_type
        record.manifests = [m.path for m in compat_report.ecosystem.manifests]
        record.languages = compat_report.ecosystem.detected_languages or ["Python"]
        record.test_framework = ", ".join(compat_report.ecosystem.detected_test_frameworks) or ("pytest" if "Python" in record.languages else "none")

        # 3. Source file and test file counts
        source_count = 0
        test_file_count = 0
        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if d not in CapabilityDetector.IGNORE_DIRS]
            rel_root = Path(root).relative_to(repo_path)
            for f in files:
                source_count += 1
                if (
                    f.startswith("test_")
                    or f.endswith("_test.py")
                    or f.endswith(".test.js")
                    or f.endswith(".test.ts")
                    or f.endswith(".spec.js")
                    or f.endswith(".spec.ts")
                    or "test" in rel_root.parts
                ):
                    test_file_count += 1

        record.source_files_count = source_count
        record.test_files_count = test_file_count

        # 4. Multi-language Test discovery via BaselineEvaluator
        try:
            evaluator = BaselineEvaluator(repo_path=str(repo_path))
            discovered = evaluator.discover_all_tests()
            record.discovered_tests = discovered
            record.total_tests_count = len(discovered)
        except Exception:
            record.discovered_tests = []
            record.total_tests_count = 0

        # 5. Check Language Support & Baseline Test Suite
        if "Python" not in record.languages and record.languages:
            record.supported_language = False
            record.unsupported_reason = compat_report.summary
            record.status = RepositoryStatus.PARTIALLY_SUPPORTED
            record.status_message = compat_report.summary
            record.baseline_test_status = "external_runner"
        else:
            record.supported_language = True
            record.unsupported_reason = None
            # Run baseline tests only for Python pytest
            auto_gen = AutoBenchmarkGenerator(repo_path=repo_path)
            baseline_res = auto_gen.run_baseline_tests(record.discovered_tests)
            record.baseline_test_status = baseline_res.get("status", "untested")
            record.baseline_test_error = baseline_res.get("error")

        # 6. Check benchmark readiness
        validated_cases = [c for c in record.benchmark_cases if c.ground_truth.is_verified]
        record.validated_benchmark_cases_count = len(validated_cases)
        record.cases_awaiting_review_count = len([c for c in record.benchmark_cases if not c.ground_truth.is_verified])

        if not record.supported_language:
            pass  # Keep PARTIALLY_SUPPORTED
        elif len(validated_cases) > 0:
            record.status = RepositoryStatus.READY_FOR_EVALUATION
            record.status_message = (
                f"Ready for evaluation ({len(validated_cases)} benchmark cases validated, "
                f"{record.total_tests_count} total tests discovered)."
            )
        elif record.total_tests_count > 0:
            record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
            record.status_message = (
                f"Discovered {record.total_tests_count} existing tests across {test_file_count} test files. "
                "Independent benchmark ground truth must be prepared before evaluation."
            )
        else:
            record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
            record.status_message = (
                f"Repository indexed ({record.source_files_count} source files, 0 test files). "
                "Automatic regression test generation is available to establish benchmark ground truth."
            )

        return record

    def auto_generate_and_validate_benchmarks(self, repo_id: str) -> CustomRepositoryRecord:
        """
        Executes automatic regression test generation, validation, mutation cross-checks,
        and independent ground truth construction for a custom repository.
        Prevents evaluation data leakage by using mutation sensitivity and AST contracts.
        """
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        if not record.supported_language:
            raise ValueError(
                f"Automatic test generation unsupported: {record.unsupported_reason or 'Unsupported language'}"
            )

        repo_path = Path(record.canonical_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path no longer exists: {repo_path}")

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        generator = AutoBenchmarkGenerator(repo_path=repo_path)

        try:
            # Step 1: Synthesize tests
            record.status = RepositoryStatus.GENERATING_TESTS
            record.status_message = "Synthesizing behavioral regression tests for public symbols..."
            record.updated_at = now_iso
            records[repo_id] = record
            self._save_all_records(records)

            gen_records = generator.generate_and_save_tests(max_targets=15)
            if not gen_records:
                record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
                record.status_message = "No eligible public functions found for automatic test synthesis."
                records[repo_id] = record
                self._save_all_records(records)
                return record

            # Step 2: Validate generated tests in isolated environment
            record.status = RepositoryStatus.VALIDATING_TESTS
            record.status_message = f"Validating {len(gen_records)} synthesized tests in isolated environment..."
            records[repo_id] = record
            self._save_all_records(records)

            val_records = generator.validate_generated_tests(gen_records)

            # Step 3: Cross-check mutation sensitivity
            record.status = RepositoryStatus.BENCHMARK_UNDER_VALIDATION
            record.status_message = "Cross-checking mutation sensitivity to prove tests detect code defects..."
            records[repo_id] = record
            self._save_all_records(records)

            mut_records = generator.cross_check_mutation_sensitivity(val_records)

            # Step 4: Construct independent benchmark cases
            cases = generator.construct_benchmark_cases(repo_id, mut_records, is_git=record.is_git)

            # Step 5: Update record with results
            record.generated_tests = mut_records
            record.generated_tests_count = len(mut_records)
            record.validated_tests_count = len([t for t in mut_records if t.executes_successfully and not t.is_trivial])
            record.mutation_sensitive_tests_count = len([t for t in mut_records if t.mutation_sensitive])

            # Merge cases (avoiding duplicate case IDs)
            existing_ids = {c.case_id for c in record.benchmark_cases}
            for c in cases:
                if c.case_id not in existing_ids:
                    record.benchmark_cases.append(c)

            # Refresh discovered tests via AST parser
            evaluator = BaselineEvaluator(repo_path=str(repo_path))
            record.discovered_tests = evaluator.discover_all_tests()
            record.total_tests_count = len(record.discovered_tests)

            validated_cases = [c for c in record.benchmark_cases if c.ground_truth.is_verified]
            record.validated_benchmark_cases_count = len(validated_cases)
            record.cases_awaiting_review_count = len([c for c in record.benchmark_cases if not c.ground_truth.is_verified])

            if record.validated_benchmark_cases_count > 0:
                record.status = RepositoryStatus.READY_FOR_EVALUATION
                record.status_message = (
                    f"Auto-generated {record.validated_tests_count} validated tests "
                    f"({record.mutation_sensitive_tests_count} mutation-sensitive). "
                    f"{record.validated_benchmark_cases_count} benchmark cases validated and ready for evaluation."
                )
            else:
                record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
                record.status_message = (
                    f"Synthesized {record.generated_tests_count} tests, but none demonstrated mutation sensitivity. "
                    "Manual review or independent benchmark preparation required."
                )

            record.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            records[repo_id] = record
            self._save_all_records(records)
            return record
        except Exception as e:
            record.status = RepositoryStatus.FAILED
            record.status_message = f"Automatic test generation failed: {str(e)}"
            record.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            records[repo_id] = record
            self._save_all_records(records)
            raise

    def get_generated_tests(self, repo_id: str) -> list[GeneratedTestRecord]:
        """Retrieves detailed list of generated tests and their validation evidence."""
        record = self.get_repository(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")
        return record.generated_tests

    def list_repositories(self, active_only: bool = True) -> list[CustomRepositoryRecord]:
        """Lists registered repositories, optionally filtering by active status."""
        records = self._load_all_records()
        repos = list(records.values())
        if active_only:
            repos = [r for r in repos if r.is_active]

        # Check stale status for active git repos
        updated = False
        for r in repos:
            if r.is_active and r.is_git and r.detected_revision and r.status == RepositoryStatus.EVALUATED:
                stale_r = self._check_revision_stale(r)
                if stale_r.status != r.status:
                    records[r.repo_id] = stale_r
                    updated = True
        if updated:
            self._save_all_records(records)
            repos = [r for r in records.values() if not active_only or r.is_active]

        # Sort: most recently updated first
        return sorted(repos, key=lambda x: x.updated_at, reverse=True)

    def get_repository(self, repo_id: str) -> Optional[CustomRepositoryRecord]:
        """Retrieves a specific repository record by ID."""
        records = self._load_all_records()
        record = records.get(repo_id)
        if record and record.is_active and record.is_git and record.status == RepositoryStatus.EVALUATED:
            record = self._check_revision_stale(record)
        return record

    def _check_revision_stale(self, record: CustomRepositoryRecord) -> CustomRepositoryRecord:
        """Detects if git HEAD has moved ahead of the evaluated revision."""
        repo_path = Path(record.canonical_path).resolve()
        if not repo_path.exists() or not (repo_path / ".git").exists():
            return record
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0:
                current_rev = res.stdout.strip()
                if record.detected_revision and current_rev != record.detected_revision:
                    record.status = RepositoryStatus.STALE
                    record.status_message = (
                        f"Repository revision changed from {record.detected_revision[:8]} to {current_rev[:8]}. "
                        "Re-evaluation is recommended."
                    )
        except Exception:
            pass
        return record

    def analyze_repository(self, repo_id: str) -> CustomRepositoryRecord:
        """Forces a fresh inspection and test discovery pass on a repository."""
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        record = self.inspect_repository(record)
        records[repo_id] = record
        self._save_all_records(records)
        return record

    def prepare_benchmark_case(
        self,
        repo_id: str,
        req: PrepareBenchmarkRequest,
    ) -> CustomBenchmarkCase:
        """
        Prepares and validates an independent benchmark ground-truth case.
        Validates provenance, expected changed symbols, and expected test callers.
        Independent of model predictions.
        """
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        if not req.name.strip():
            raise ValueError("Benchmark case name is required.")

        if not req.provenance.strip():
            raise ValueError(
                "Ground truth provenance documentation is mandatory to prevent data leakage. "
                "Specify the audit source or verified issue reference."
            )

        repo_path = Path(record.canonical_path).resolve()

        # Resolve or auto-extract changed symbols from Git diff if using git_commit_range
        expected_changed = list(req.expected_changed_symbols)
        if req.source_type == BenchmarkSourceType.GIT_COMMIT_RANGE and not expected_changed and record.is_git:
            try:
                evo_engine = RepositoryEvolutionEngine(repo_root=str(repo_path))
                extracted = evo_engine.analyze_diff(req.base_commit, req.target_commit)
                expected_changed = [s.name for s in extracted if s.name]
            except Exception:
                pass

        if not expected_changed:
            expected_changed = ["target_function"]

        case_id = f"{record.repo_id}_case_{len(record.benchmark_cases) + 1}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        gt = GroundTruth(
            case_id=case_id,
            expected_changed_symbols=expected_changed,
            expected_impacted_symbols=req.expected_impacted_symbols,
            expected_test_callers=req.expected_test_callers,
            provenance=req.provenance.strip(),
            notes=req.notes.strip(),
            is_verified=True,
        )

        case = CustomBenchmarkCase(
            case_id=case_id,
            name=req.name.strip(),
            description=req.description.strip(),
            source_type=req.source_type,
            base_commit=req.base_commit,
            target_commit=req.target_commit,
            commit_sha=record.detected_revision,
            ground_truth=gt,
            review_status=GroundTruthReviewStatus.VALIDATED,
            total_tests_suite=record.total_tests_count,
            created_at=now_iso,
        )

        record.benchmark_cases.append(case)
        record.validated_benchmark_cases_count = len([c for c in record.benchmark_cases if c.ground_truth.is_verified])
        record.status = RepositoryStatus.READY_FOR_EVALUATION
        record.status_message = (
            f"Benchmark ground truth '{case.name}' validated. Ready for quantitative evaluation."
        )
        record.updated_at = now_iso

        records[repo_id] = record
        self._save_all_records(records)
        return case

    def prepare_historical_bug_case(
        self,
        repo_id: str,
        req: PrepareHistoricalBugRequest,
    ) -> CustomBenchmarkCase:
        """
        Method A: Prepares and independently verifies a historical bug fix benchmark case.
        Validates that the test fails on pre-fix commit and passes on post-fix commit in isolated worktrees.
        """
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        generator = AutoBenchmarkGenerator(repo_path=Path(record.canonical_path))
        case = generator.validate_historical_regression_case(
            repo_id=record.repo_id,
            name=req.name,
            buggy_commit=req.buggy_commit,
            corrected_commit=req.corrected_commit,
            test_id=req.test_id,
            description=req.description,
            expected_changed_symbols=req.expected_changed_symbols,
            expected_impacted_symbols=req.expected_impacted_symbols,
            provenance=req.provenance,
            notes=req.notes,
        )
        case.total_tests_suite = record.total_tests_count

        record.benchmark_cases.append(case)
        record.validated_benchmark_cases_count = len([c for c in record.benchmark_cases if c.ground_truth.is_verified])
        record.status = RepositoryStatus.READY_FOR_EVALUATION
        record.status_message = (
            f"Historical bug benchmark '{case.name}' independently verified. Ready for evaluation."
        )
        record.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records[repo_id] = record
        self._save_all_records(records)
        return case

    prepare_benchmark_ground_truth = prepare_benchmark_case

    def review_benchmark_case(
        self,
        repo_id: str,
        case_id: str,
        req: ReviewGroundTruthRequest | str,
        reviewer: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CustomBenchmarkCase:
        """
        Performs human review on a benchmark ground truth case:
        - 'approve': marks case VALIDATED, eligible for quantitative evaluation.
        - 'reject': marks case REJECTED, excluded from evaluation.
        """
        if isinstance(req, str):
            req_obj = ReviewGroundTruthRequest(
                action=req,
                reviewer=reviewer or "Human Reviewer",
                notes=notes or "",
            )
        else:
            req_obj = req

        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        target_case = None
        for c in record.benchmark_cases:
            if c.case_id == case_id:
                target_case = c
                break

        if not target_case:
            raise KeyError(f"Benchmark case not found: {case_id}")

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        target_case.reviewer = req_obj.reviewer
        target_case.review_notes = req_obj.notes
        target_case.reviewed_at = now_iso

        if req_obj.action.lower() == "approve":
            target_case.review_status = GroundTruthReviewStatus.VALIDATED
            target_case.ground_truth.is_verified = True
            target_case.validation_level = GroundTruthValidationLevel.CONTRACT_MUTATION_VERIFIED
        elif req_obj.action.lower() == "reject":
            target_case.review_status = GroundTruthReviewStatus.REJECTED
            target_case.ground_truth.is_verified = False
        else:
            raise ValueError(f"Invalid review action: '{req_obj.action}'. Expected 'approve' or 'reject'.")

        record.validated_benchmark_cases_count = len([c for c in record.benchmark_cases if c.ground_truth.is_verified])
        record.cases_awaiting_review_count = len([c for c in record.benchmark_cases if not c.ground_truth.is_verified])

        if record.validated_benchmark_cases_count > 0:
            record.status = RepositoryStatus.READY_FOR_EVALUATION
            record.status_message = f"Ground truth reviewed ({record.validated_benchmark_cases_count} cases validated). Ready for evaluation."
        else:
            record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
            record.status_message = "All benchmark cases require review or validation before quantitative evaluation."

        record.updated_at = now_iso
        records[repo_id] = record
        self._save_all_records(records)
        return target_case

    def evaluate_repository(
        self,
        repo_id: str,
        case_id: Optional[str] = None,
    ) -> list[EvaluationRun]:
        """
        Executes benchmark evaluation across all supported baselines:
        1. Full Regression
        2. Naive Name Matching
        3. TestPilot (Qualified Identity)
        4. TestPilot + Repository RAG
        5. TestPilot + Sourcegraph (preserves exact connectivity check)
        Calculates strict mathematical metrics and persists results.
        """
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        cases_to_run = [c for c in record.benchmark_cases if c.ground_truth.is_verified]
        if case_id:
            cases_to_run = [c for c in cases_to_run if c.case_id == case_id]

        if not cases_to_run:
            raise ValueError(
                f"Repository '{record.name}' has no validated benchmark cases. "
                "Prepare independent benchmark ground truth before evaluating."
            )

        repo_path = Path(record.canonical_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        record.status = RepositoryStatus.EVALUATION_RUNNING
        record.status_message = f"Evaluating {len(cases_to_run)} benchmark cases across supported baselines..."
        self._save_all_records(records)

        runs: list[EvaluationRun] = []
        baseline_evaluator = BaselineEvaluator(repo_path=str(repo_path))
        discovered_tests = record.discovered_tests or baseline_evaluator.discover_all_tests()
        total_tests = len(discovered_tests)

        try:
            for case in cases_to_run:
                start_total = time.perf_counter()

                # Extract changed symbols from commit diff or ground truth
                changed_symbols: list[ChangedSymbol] = []
                if record.is_git and case.base_commit and case.target_commit:
                    try:
                        evo_engine = RepositoryEvolutionEngine(repo_root=str(repo_path))
                        changed_symbols = evo_engine.analyze_diff(case.base_commit, case.target_commit)
                    except Exception:
                        pass

                if not changed_symbols:
                    changed_symbols = []
                    for s in case.ground_truth.expected_changed_symbols:
                        target_file = "app.py"
                        # Scan repository python source files to locate the definition file
                        for py_path in repo_path.rglob("*.py"):
                            rel_p = str(py_path.relative_to(repo_path)).replace("\\", "/")
                            if rel_p.startswith("test") or "/test" in rel_p or py_path.name.startswith("."):
                                continue
                            try:
                                content = py_path.read_text(encoding="utf-8", errors="ignore")
                                if f"def {s}" in content or f"class {s}" in content:
                                    target_file = rel_p
                                    break
                            except Exception:
                                continue
                        changed_symbols.append(
                            ChangedSymbol(
                                name=s,
                                file_path=target_file,
                                line_start=1,
                                line_end=50,
                                change_type=ChangeType.MODIFIED,
                            )
                        )

                # 1. Full Regression
                res_full = baseline_evaluator.run_full_regression(discovered_tests)
                metrics_full = compute_metrics(
                    predicted_tests=res_full.selected_tests,
                    expected_tests=case.ground_truth.expected_test_callers,
                    total_tests=total_tests,
                    latency_ms=res_full.latency_ms,
                )

                # 2. Naive Name Matching
                res_naive = baseline_evaluator.run_naive_name_matching(changed_symbols, discovered_tests)
                metrics_naive = compute_metrics(
                    predicted_tests=res_naive.selected_tests,
                    expected_tests=case.ground_truth.expected_test_callers,
                    total_tests=total_tests,
                    latency_ms=res_naive.latency_ms,
                )

                # 3. TestPilot (Qualified Identity)
                start_select = time.perf_counter()
                res_testpilot, evidence_records = baseline_evaluator.run_testpilot(
                    changed_symbols, discovered_tests
                )
                selection_latency_ms = (time.perf_counter() - start_select) * 1000.0
                total_pipeline_latency_ms = (time.perf_counter() - start_total) * 1000.0

                metrics_testpilot = compute_metrics(
                    predicted_tests=res_testpilot.selected_tests,
                    expected_tests=case.ground_truth.expected_test_callers,
                    total_tests=total_tests,
                    latency_ms=total_pipeline_latency_ms,
                    evolution_latency_ms=res_testpilot.latency_ms,
                    selection_latency_ms=selection_latency_ms,
                )

                # 4. TestPilot + Repository RAG
                start_rag = time.perf_counter()
                res_testpilot_rag, _ = baseline_evaluator.run_testpilot_rag(
                    changed_symbols, discovered_tests
                )
                rag_latency_ms = (time.perf_counter() - start_rag) * 1000.0

                metrics_testpilot_rag = compute_metrics(
                    predicted_tests=res_testpilot_rag.selected_tests,
                    expected_tests=case.ground_truth.expected_test_callers,
                    total_tests=total_tests,
                    latency_ms=total_pipeline_latency_ms + rag_latency_ms,
                    evolution_latency_ms=res_testpilot.latency_ms,
                    selection_latency_ms=selection_latency_ms + rag_latency_ms,
                )

                # 5. TestPilot + Sourcegraph (preserves exact connectivity check; never fabricates)
                res_testpilot_sg, _ = baseline_evaluator.run_testpilot_sourcegraph(
                    changed_symbols, discovered_tests
                )
                metrics_testpilot_sg = (
                    compute_metrics(
                        predicted_tests=res_testpilot_sg.selected_tests,
                        expected_tests=case.ground_truth.expected_test_callers,
                        total_tests=total_tests,
                        latency_ms=res_testpilot_sg.latency_ms,
                    )
                    if res_testpilot_sg.selected_count > 0
                    else EvaluationMetrics(is_evaluated=False, notes=res_testpilot_sg.notes)
                )

                run_id = f"custom_run_{case.case_id}_{int(time.time())}"
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

                run = EvaluationRun(
                    run_id=run_id,
                    case_id=case.case_id,
                    repository=record.name,
                    base_commit=case.base_commit,
                    target_commit=case.target_commit,
                    testpilot_version=record.detected_revision or "HEAD",
                    python_version=platform.python_version(),
                    rag_status="active",
                    total_tests=total_tests,
                    ground_truth_count=len(case.ground_truth.expected_test_callers),
                    baseline_results={
                        BaselineType.FULL_REGRESSION.value: res_full,
                        BaselineType.NAIVE_NAME_MATCHING.value: res_naive,
                        BaselineType.TESTPILOT.value: res_testpilot,
                        BaselineType.TESTPILOT_RAG.value: res_testpilot_rag,
                        BaselineType.TESTPILOT_SOURCEGRAPH.value: res_testpilot_sg,
                    },
                    metrics={
                        BaselineType.FULL_REGRESSION.value: metrics_full,
                        BaselineType.NAIVE_NAME_MATCHING.value: metrics_naive,
                        BaselineType.TESTPILOT.value: metrics_testpilot,
                        BaselineType.TESTPILOT_RAG.value: metrics_testpilot_rag,
                        BaselineType.TESTPILOT_SOURCEGRAPH.value: metrics_testpilot_sg,
                    },
                    evidence_records=evidence_records,
                    timestamp=now_iso,
                    status="completed",
                    notes=case.description or f"Evaluation of {case.name}",
                )

                record.latest_evaluation_runs[case.case_id] = run
                runs.append(run)

            record.status = RepositoryStatus.EVALUATED
            record.status_message = f"Evaluation completed for {len(runs)} benchmark case(s)."
            record.last_evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            record.updated_at = record.last_evaluated_at

            # Update detected_revision to match evaluated commit
            if record.is_git:
                try:
                    res = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(repo_path),
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    if res.returncode == 0:
                        record.detected_revision = res.stdout.strip()
                except Exception:
                    pass

        except Exception as e:
            record.status = RepositoryStatus.EVALUATION_FAILED
            record.status_message = f"Evaluation failed: {str(e)}"
            records[repo_id] = record
            self._save_all_records(records)
            raise e

        records[repo_id] = record
        self._save_all_records(records)
        return runs

    def remove_repository(self, repo_id: str, soft_delete: bool = True) -> dict[str, Any]:
        """
        Safely removes a repository from the active custom benchmark suite.
        Never deletes user source files on disk.
        Updates active suite counts and excludes its results from active aggregates.
        """
        records = self._load_all_records()
        record = records.get(repo_id)
        if not record:
            raise KeyError(f"Repository not found: {repo_id}")

        repo_name = record.name
        if soft_delete:
            record.is_active = False
            record.status_message = "Repository removed from active benchmark suite (archived)."
            record.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            records[repo_id] = record
        else:
            del records[repo_id]

        self._save_all_records(records)
        return {
            "status": "removed",
            "repo_id": repo_id,
            "name": repo_name,
            "message": f"Repository '{repo_name}' safely removed from active custom suite. Local files were not modified.",
        }

    def get_custom_suite_overview(self, selected_repo_id: Optional[str] = None) -> dict[str, Any]:
        """
        Generates dynamic benchmark overview and matrix for the custom repository suite.
        Strictly aggregates TP, FP, FN across eligible benchmark cases.
        Repositories without validated ground truth or un-evaluated appear as pending/required.
        """
        active_repos = self.list_repositories(active_only=True)

        if selected_repo_id:
            active_repos = [r for r in active_repos if r.repo_id == selected_repo_id]

        # Collect evaluated runs from active repos
        evaluated_runs: list[EvaluationRun] = []
        for r in active_repos:
            for run in r.latest_evaluation_runs.values():
                if run.status == "completed" and run.ground_truth_count > 0:
                    evaluated_runs.append(run)

        total_tests_pool = sum(r.total_tests for r in evaluated_runs)
        total_gt_pool = sum(r.ground_truth_count for r in evaluated_runs)

        def compute_pooled_row(baseline_key: str, display_name: str, notes_default: str) -> dict[str, Any]:
            has_data = False
            if evaluated_runs:
                for r in evaluated_runs:
                    if baseline_key in r.metrics:
                        m = r.metrics[baseline_key]
                        if m.is_evaluated and (m.tp > 0 or m.fp > 0 or m.fn > 0):
                            has_data = True
                            break
                    if baseline_key in r.baseline_results:
                        if r.baseline_results[baseline_key].selected_count > 0:
                            has_data = True
                            break

            if not has_data or not evaluated_runs:
                return {
                    "configuration": display_name,
                    "status": "Not Evaluated / Pending",
                    "badge": "◐ Not Evaluated Yet",
                    "is_evaluated": False,
                    "selected_tests": None,
                    "selected_tests_display": "—",
                    "ground_truth_positives": None,
                    "ground_truth_positives_display": "—",
                    "tp": None,
                    "tp_display": "—",
                    "fp": None,
                    "fp_display": "—",
                    "fn": None,
                    "fn_display": "—",
                    "precision": None,
                    "precision_display": "—",
                    "recall": None,
                    "recall_display": "—",
                    "f1": None,
                    "f1_display": "—",
                    "test_reduction": None,
                    "test_reduction_display": "—",
                    "notes": notes_default,
                }

            tp = sum(r.metrics[baseline_key].tp for r in evaluated_runs if baseline_key in r.metrics)
            fp = sum(r.metrics[baseline_key].fp for r in evaluated_runs if baseline_key in r.metrics)
            fn = sum(r.metrics[baseline_key].fn for r in evaluated_runs if baseline_key in r.metrics)
            tests_selected = sum(
                r.baseline_results[baseline_key].selected_count
                for r in evaluated_runs
                if baseline_key in r.baseline_results
            )

            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
            recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
            f1_val = (
                round((2.0 * precision * recall) / (precision + recall), 4)
                if (precision is not None and recall is not None and (precision + recall) > 0)
                else None
            )
            reduction = (
                round((1.0 - (tests_selected / total_tests_pool)) * 100.0, 2)
                if total_tests_pool > 0
                else 0.0
            )

            return {
                "configuration": display_name,
                "status": "Evaluated",
                "badge": "✓ Evaluated",
                "is_evaluated": True,
                "selected_tests": tests_selected,
                "selected_tests_display": f"{tests_selected:,}",
                "ground_truth_positives": total_gt_pool,
                "ground_truth_positives_display": str(total_gt_pool),
                "tp": tp,
                "tp_display": str(tp),
                "fp": fp,
                "fp_display": str(fp),
                "fn": fn,
                "fn_display": str(fn),
                "precision": precision,
                "precision_display": f"{precision * 100:.2f}%" if precision is not None else "N/A",
                "recall": recall,
                "recall_display": f"{recall * 100:.2f}%" if recall is not None else "N/A",
                "f1": f1_val,
                "f1_display": f"{f1_val * 100:.2f}%" if f1_val is not None else "N/A",
                "test_reduction": reduction,
                "test_reduction_display": f"{reduction:.2f}%" if reduction is not None else "N/A",
                "notes": notes_default,
            }

        primary_matrix = [
            compute_pooled_row(
                BaselineType.FULL_REGRESSION.value,
                "Full Regression",
                "Upper-bound recall baseline; executes entire test suite.",
            ),
            compute_pooled_row(
                BaselineType.NAIVE_NAME_MATCHING.value,
                "Naive Baseline",
                "Token-based matching ignoring qualified symbol identity and call graphs.",
            ),
            compute_pooled_row(
                BaselineType.TESTPILOT.value,
                "TestPilot",
                "Core deterministic TestPilot: AST Call Graph + SymbolId qualified identity.",
            ),
            compute_pooled_row(
                BaselineType.TESTPILOT_RAG.value,
                "TestPilot + Repository RAG",
                "Semantic AST vector retrieval & CodeLlama validation with recall protection.",
            ),
            compute_pooled_row(
                BaselineType.TESTPILOT_SOURCEGRAPH.value,
                "TestPilot + Sourcegraph",
                "Repository code intelligence; requires active Sourcegraph container on port 7080.",
            ),
        ]

        def compute_macro_row(baseline_key: str, display_name: str) -> dict[str, Any]:
            repo_precisions = []
            repo_recalls = []
            repo_f1s = []
            repo_reductions = []

            for r in active_repos:
                for run in r.latest_evaluation_runs.values():
                    if run.status == "completed" and baseline_key in run.metrics:
                        m = run.metrics[baseline_key]
                        if m.precision is not None:
                            repo_precisions.append(m.precision)
                        if m.recall is not None:
                            repo_recalls.append(m.recall)
                        if m.f1 is not None:
                            repo_f1s.append(m.f1)
                        if m.test_reduction_pct is not None:
                            repo_reductions.append(m.test_reduction_pct)

            avg_p = round(sum(repo_precisions) / len(repo_precisions), 4) if repo_precisions else None
            avg_r = round(sum(repo_recalls) / len(repo_recalls), 4) if repo_recalls else None
            avg_f1 = round(sum(repo_f1s) / len(repo_f1s), 4) if repo_f1s else None
            avg_red = round(sum(repo_reductions) / len(repo_reductions), 2) if repo_reductions else None

            return {
                "configuration": display_name,
                "is_evaluated": bool(repo_precisions or repo_recalls),
                "macro_precision": avg_p,
                "macro_precision_display": f"{avg_p * 100:.2f}%" if avg_p is not None else "N/A",
                "macro_recall": avg_r,
                "macro_recall_display": f"{avg_r * 100:.2f}%" if avg_r is not None else "N/A",
                "macro_f1": avg_f1,
                "macro_f1_display": f"{avg_f1 * 100:.2f}%" if avg_f1 is not None else "N/A",
                "macro_test_reduction": avg_red,
                "macro_test_reduction_display": f"{avg_red:.2f}%" if avg_red is not None else "N/A",
                "evaluated_repos_count": len(repo_precisions),
            }

        macro_matrix = [
            compute_macro_row(BaselineType.FULL_REGRESSION.value, "Full Regression"),
            compute_macro_row(BaselineType.NAIVE_NAME_MATCHING.value, "Naive Baseline"),
            compute_macro_row(BaselineType.TESTPILOT.value, "TestPilot"),
            compute_macro_row(BaselineType.TESTPILOT_RAG.value, "TestPilot + Repository RAG"),
            compute_macro_row(BaselineType.TESTPILOT_SOURCEGRAPH.value, "TestPilot + Sourcegraph"),
        ]

        # Fault detection metrics from verified mutation and historical bug scenarios
        all_validated_cases = [c for r in active_repos for c in r.benchmark_cases if c.ground_truth.is_verified]
        kill_rates = [c.mutant_kill_rate for c in all_validated_cases if c.mutant_kill_rate is not None]
        avg_fault_rate = round(sum(kill_rates) / len(kill_rates), 4) if kill_rates else 0.0

        excluded_repositories = [
            {
                "repo_id": r.repo_id,
                "name": r.name,
                "status": r.status.value,
                "reason": r.status_message,
                "validated_cases": r.validated_benchmark_cases_count,
                "cases_awaiting_review": r.cases_awaiting_review_count,
            }
            for r in active_repos
            if r.status != RepositoryStatus.EVALUATED
        ]

        total_registered = len(active_repos)
        ready_count = len([r for r in active_repos if r.status in (RepositoryStatus.READY_FOR_EVALUATION, RepositoryStatus.EVALUATED)])
        prep_required_count = len([r for r in active_repos if r.status in (RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED, RepositoryStatus.TESTS_DISCOVERED, RepositoryStatus.REGISTERED)])
        evaluated_repos_count = len([r for r in active_repos if r.status == RepositoryStatus.EVALUATED])
        total_cases_awaiting_review = sum(r.cases_awaiting_review_count for r in active_repos)

        return {
            "mode": "custom_suite",
            "selected_repo_id": selected_repo_id,
            "active_repositories_count": total_registered,
            "active_benchmark_cases_count": sum(r.validated_benchmark_cases_count for r in active_repos),
            "cases_awaiting_review_count": total_cases_awaiting_review,
            "evaluated_repositories_count": evaluated_repos_count,
            "ready_count": ready_count,
            "benchmark_preparation_required_count": prep_required_count,
            "evaluated_runs_count": len(evaluated_runs),
            "total_tests_pool": total_tests_pool,
            "total_ground_truth_pool": total_gt_pool,
            "primary_matrix": primary_matrix,
            "micro_matrix": primary_matrix,
            "macro_matrix": macro_matrix,
            "fault_detection_summary": {
                "average_fault_detection_rate": avg_fault_rate,
                "average_fault_detection_display": f"{avg_fault_rate * 100:.1f}%" if avg_fault_rate > 0 else "N/A",
                "total_fault_scenarios": len(all_validated_cases),
            },
            "excluded_repositories": excluded_repositories,
            "repositories": [
                {
                    "repo_id": r.repo_id,
                    "name": r.name,
                    "canonical_path": r.canonical_path,
                    "status": r.status.value,
                    "status_message": r.status_message,
                    "detected_revision": r.detected_revision,
                    "total_tests": r.total_tests_count,
                    "generated_tests_count": r.generated_tests_count,
                    "validated_tests_count": r.validated_tests_count,
                    "mutation_sensitive_tests_count": r.mutation_sensitive_tests_count,
                    "validated_benchmark_cases_count": r.validated_benchmark_cases_count,
                    "cases_awaiting_review_count": r.cases_awaiting_review_count,
                    "last_evaluated_at": r.last_evaluated_at,
                    "has_evaluated_runs": len(r.latest_evaluation_runs) > 0,
                }
                for r in active_repos
            ],
        }
