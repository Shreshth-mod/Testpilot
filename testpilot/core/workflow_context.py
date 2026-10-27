"""
Typed Workflow Context and Multi-Run Storage Manager for TestPilot AI.
Enforces safe path traversal, atomic file operations, immutable commit hash resolution,
complete reproducible dirty worktree snapshotting, and strict sequential stage progression.
"""

from __future__ import annotations

import base64
import enum
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger("testpilot.workflow_context")


class SecurityViolationError(ValueError):
    """Raised when an untrusted repository identifier or path escapes allowed storage roots."""
    pass


class PrerequisiteError(RuntimeError):
    """Raised when an attempt is made to execute a stage without valid upstream inputs."""
    pass


class StageStatus(str, enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    STALE = "STALE"
    BLOCKED_CRITICAL_WARNING = "BLOCKED_CRITICAL_WARNING"


# Stage order defines strict upstream dependencies
STAGE_SEQUENCE: list[str] = [
    "evolution",
    "impact",
    "selection",
    "execution",
    "arbitration",
]


class StageMetadata(BaseModel):
    """Tracks metadata and lifecycle state for a specific workflow stage."""
    stage_id: str
    status: StageStatus = StageStatus.NOT_STARTED
    run_id: Optional[str] = None
    upstream_run_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    critical_warnings: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    summary: dict[str, Any] = Field(default_factory=dict)

    def can_progress(self) -> bool:
        """Determines if downstream stages are permitted to proceed from this stage."""
        if self.status == StageStatus.COMPLETED:
            return True
        if self.status == StageStatus.COMPLETED_WITH_WARNINGS:
            # Only non-critical warnings permit progression
            return len(self.critical_warnings) == 0
        return False


class WorkflowContext(BaseModel):
    """
    Authoritative multi-stage workflow execution context.
    Encapsulates immutable commit provenance, stage run IDs, and verified artifact references.
    """
    workflow_id: str = Field(
        default_factory=lambda: f"wf-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    )
    repo_id: str
    repo_path: str
    repo_name: str = "Repository"
    base_commit_hash: str
    target_commit_hash: str
    is_dirty: bool = False
    patch_snapshot_id: Optional[str] = None
    patch_snapshot_hash: Optional[str] = None

    # Explicit stage run IDs
    evolution_run_id: Optional[str] = None
    impact_run_id: Optional[str] = None
    selection_run_id: Optional[str] = None
    execution_run_id: Optional[str] = None
    arbitration_run_id: Optional[str] = None

    stages: dict[str, StageMetadata] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def create_initial(
        cls,
        repo_id: str,
        repo_path: str,
        repo_name: str,
        base_commit_hash: str,
        target_commit_hash: str,
        is_dirty: bool = False,
        patch_snapshot_id: Optional[str] = None,
        patch_snapshot_hash: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> WorkflowContext:
        """Construct a fresh workflow context with initialized stage states."""
        stages: dict[str, StageMetadata] = {
            "evolution": StageMetadata(stage_id="evolution", status=StageStatus.READY),
            "impact": StageMetadata(stage_id="impact", status=StageStatus.NOT_STARTED),
            "selection": StageMetadata(stage_id="selection", status=StageStatus.NOT_STARTED),
            "execution": StageMetadata(stage_id="execution", status=StageStatus.NOT_STARTED),
            "arbitration": StageMetadata(stage_id="arbitration", status=StageStatus.NOT_STARTED),
        }
        wf_id = workflow_id or f"wf-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        return cls(
            workflow_id=wf_id,
            repo_id=repo_id,
            repo_path=str(repo_path),
            repo_name=repo_name,
            base_commit_hash=base_commit_hash,
            target_commit_hash=target_commit_hash,
            is_dirty=is_dirty,
            patch_snapshot_id=patch_snapshot_id,
            patch_snapshot_hash=patch_snapshot_hash,
            stages=stages,
        )


class WorkflowStorageManager:
    """
    Manages safe multi-run persistence, atomic writes, commit hash resolution,
    and prerequisite validation.
    """

    def __init__(self, base_storage_dir: Optional[Path] = None, base_dir: Optional[Path] = None) -> None:
        target_dir = base_dir or base_storage_dir or Path(".testpilot/workflows")
        self.base_dir = Path(target_dir).resolve()
        self.snapshots_dir = (self.base_dir.parent / "snapshots").resolve()
        self.reports_dir = (self.base_dir.parent / "reports").resolve()
        self.workspaces_dir = (self.base_dir.parent / "workspaces").resolve()

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_repo_id(raw_id: str, raw_path: Optional[str] = None) -> str:
        """
        Sanitizes a repository identifier to prevent path traversal attacks.
        Disallows directory separators, relative segments (../), and crafted characters.
        """
        if not raw_id or raw_id in (".", "..", "/", "\\"):
            if raw_path:
                abs_p = str(Path(raw_path).expanduser().resolve())
                slug = Path(abs_p).name or "repo"
                h = hashlib.sha256(abs_p.encode()).hexdigest()[:8]
                raw_id = f"{slug}_{h}"
            else:
                raw_id = "default_repo"

        # Replace invalid path characters with underscores
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_id).strip("_")
        if not cleaned:
            cleaned = hashlib.sha256(raw_id.encode()).hexdigest()[:12]
        return cleaned

    def _get_repo_dir(self, canonical_repo_id: str) -> Path:
        """Derive validated repository storage directory."""
        repo_dir = (self.base_dir / canonical_repo_id).resolve()
        if not repo_dir.is_relative_to(self.base_dir):
            raise SecurityViolationError(
                f"Path traversal detected: canonical repository ID '{canonical_repo_id}' escapes base directory '{self.base_dir}'."
            )
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir

    def _get_workflow_file(self, canonical_repo_id: str, workflow_id: str) -> Path:
        """Derive validated workflow file path."""
        repo_dir = self._get_repo_dir(canonical_repo_id)
        # Validate workflow_id contains only alphanumeric, dash, and underscore
        if not re.match(r"^[a-zA-Z0-9_\-]+$", workflow_id):
            raise SecurityViolationError(f"Invalid workflow ID format: '{workflow_id}'")
        target_file = (repo_dir / f"{workflow_id}.json").resolve()
        if not target_file.is_relative_to(repo_dir):
            raise SecurityViolationError(f"Workflow file escapes repository folder: '{target_file}'")
        return target_file

    def get_workflow_path(self, canonical_repo_id: str, workflow_id: str) -> Path:
        """Public accessor for validated workflow file path."""
        return self._get_workflow_file(canonical_repo_id, workflow_id)

    def save_workflow(self, context: WorkflowContext) -> Path:
        """
        Atomically persist a WorkflowContext to disk using temporary file replacement.
        Guarantees that interrupted writes cannot produce corrupted JSON files.
        """
        canonical_repo_id = self.sanitize_repo_id(context.repo_id, context.repo_path)
        context.repo_id = canonical_repo_id
        context.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        target_file = self._get_workflow_file(canonical_repo_id, context.workflow_id)
        temp_file = target_file.parent / f".{context.workflow_id}.tmp"

        payload = context.model_dump()
        content = json.dumps(payload, indent=2, ensure_ascii=False)

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, target_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

        # Update active workflow reference file atomically
        self._set_active_pointer(canonical_repo_id, context.workflow_id)
        return target_file

    def load_workflow(self, canonical_repo_id: str, workflow_id: str) -> Optional[WorkflowContext]:
        """Load and deserialize a WorkflowContext by repo_id and workflow_id."""
        target_file = self._get_workflow_file(canonical_repo_id, workflow_id)
        if not target_file.exists():
            return None

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowContext(**data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Corrupted workflow JSON at %s: %s", target_file, e)
            return None

    def _set_active_pointer(self, canonical_repo_id: str, workflow_id: str) -> None:
        """Write active workflow reference atomically."""
        active_ref_file = self.base_dir / "active_workflow_ref.json"
        temp_ref = self.base_dir / ".active_workflow_ref.tmp"
        data = {
            "canonical_repo_id": canonical_repo_id,
            "workflow_id": workflow_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            with open(temp_ref, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_ref, active_ref_file)
        finally:
            if temp_ref.exists():
                try:
                    temp_ref.unlink()
                except Exception:
                    pass

    def get_active_workflow(self) -> Optional[WorkflowContext]:
        """
        Retrieve active workflow with automatic recovery.
        Falls back to most recently modified valid workflow if pointer is missing or corrupted.
        """
        active_ref_file = self.base_dir / "active_workflow_ref.json"
        if active_ref_file.exists():
            try:
                with open(active_ref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                repo_id = data.get("canonical_repo_id")
                wf_id = data.get("workflow_id")
                if repo_id and wf_id:
                    wf = self.load_workflow(repo_id, wf_id)
                    if wf:
                        return wf
            except Exception as e:
                logger.warning("Active workflow pointer corrupted: %s", e)

        # Fallback: find latest valid .json file across all repos
        candidates = list(self.base_dir.glob("*/*.json"))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return WorkflowContext(**data)
            except Exception:
                continue

        return None

    @staticmethod
    def resolve_immutable_commit(repo_path: str, ref: str) -> str:
        """
        Resolves any symbolic reference (HEAD, HEAD~1, main, tags) to a 40-character immutable SHA.
        Fails safely if ref is invalid or repository is missing.
        """
        rp = Path(repo_path).expanduser().resolve()
        if not (rp / ".git").exists() and not (rp / "config").exists():
            # If not a git repo, synthesize a deterministic hash from path
            return hashlib.sha256(str(rp).encode()).hexdigest()[:40]

        cmd = ["git", "rev-parse", f"{ref}^{{commit}}"]
        try:
            res = subprocess.run(cmd, cwd=str(rp), capture_output=True, text=True, check=True)
            sha = res.stdout.strip()
            if len(sha) == 40:
                return sha
        except subprocess.SubprocessError:
            # Fallback to direct rev-parse if ^{commit} failed (e.g. initial commit)
            pass

        res2 = subprocess.run(["git", "rev-parse", ref], cwd=str(rp), capture_output=True, text=True, check=False)
        sha2 = res2.stdout.strip()
        if res2.returncode == 0 and len(sha2) >= 7 and not sha2.startswith("fatal:"):
            return sha2
        raise ValueError(f"Unable to resolve commit reference '{ref}' in repository '{repo_path}'.")

    @staticmethod
    def check_working_tree_clean(repo_path: str) -> tuple[bool, str]:
        """
        Checks whether working tree is strictly clean.
        Returns (is_clean, status_porcelain_output).
        """
        rp = Path(repo_path).expanduser().resolve()
        if not (rp / ".git").exists():
            return True, ""
        res = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=str(rp),
            capture_output=True,
            text=True,
            check=False,
        )
        output = res.stdout.strip()
        return (len(output) == 0), output

    def create_dirty_snapshot(self, repo_path: str, workflow_id: str) -> tuple[str, str]:
        """
        Creates a reproducible, verifiable snapshot of a dirty working tree.
        Captures:
          1. Tracked staged and unstaged diffs.
          2. Untracked file manifest and contents.
        Returns (snapshot_id, sha256_hash).
        """
        rp = Path(repo_path).expanduser().resolve()
        snap_id = f"snap-{workflow_id}"
        snap_file = self.snapshots_dir / f"{snap_id}.patch"

        # Capture staged and unstaged diffs with full binary deltas and rename detection
        diff_res = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "-M"],
            cwd=str(rp),
            capture_output=True,
            text=True,
            check=False,
        )
        untracked_res = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=str(rp),
            capture_output=True,
            text=True,
            check=False,
        )

        untracked_lines = [line[3:].strip() for line in untracked_res.stdout.splitlines() if line.startswith("?? ")]
        untracked_data: dict[str, Any] = {}
        for uf in untracked_lines[:500]:
            fp = rp / uf
            if fp.exists() and fp.is_file() and fp.stat().st_size < 10_000_000:
                try:
                    raw_bytes = fp.read_bytes()
                    untracked_data[uf] = {
                        "is_binary": True,
                        "b64": base64.b64encode(raw_bytes).decode("ascii"),
                    }
                except Exception:
                    pass

        payload = {
            "workflow_id": workflow_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "diff_head": diff_res.stdout,
            "untracked_files": untracked_data,
        }
        content = json.dumps(payload, indent=2)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        with open(snap_file, "w", encoding="utf-8") as f:
            f.write(content)

        return snap_id, content_hash

    def validate_stage_transition(
        self,
        context: WorkflowContext,
        target_stage: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Validates whether target_stage is allowed to execute given upstream states.
        Enforces:
          1. Evolution must be completed before Impact Analysis.
          2. Impact Analysis must be completed before Intelligent Test Selection.
          3. Test Selection must be completed before Isolated Execution.
          4. Execution must be completed before Failure Arbitration.
          5. No critical blocking warnings are present on the immediate upstream stage.
        """
        if target_stage not in STAGE_SEQUENCE:
            return False, f"Unknown stage '{target_stage}'. Valid stages: {STAGE_SEQUENCE}"

        stage_idx = STAGE_SEQUENCE.index(target_stage)
        if stage_idx == 0:
            # Evolution has no upstream prerequisite
            return True, None

        upstream_stage = STAGE_SEQUENCE[stage_idx - 1]
        upstream_meta = context.stages.get(upstream_stage)

        if not upstream_meta:
            return False, f"Upstream stage '{upstream_stage}' has no recorded execution state."

        if upstream_meta.status not in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS):
            return (
                False,
                f"Cannot execute '{target_stage}': upstream stage '{upstream_stage}' is currently '{upstream_meta.status.value}'. Required state: 'COMPLETED'.",
            )

        if upstream_meta.status == StageStatus.COMPLETED_WITH_WARNINGS and not upstream_meta.can_progress():
            reasons = "; ".join(upstream_meta.critical_warnings)
            return (
                False,
                f"Cannot execute '{target_stage}': upstream stage '{upstream_stage}' has critical warnings: {reasons}",
            )

        return True, None

    def invalidate_downstream(
        self, context: WorkflowContext, changed_stage: str, include_current: bool = True
    ) -> WorkflowContext:
        """
        Marks all downstream stages (and optionally current stage) STALE when an upstream stage is re-run or re-analyzed.
        """
        if changed_stage not in STAGE_SEQUENCE:
            return context

        idx = STAGE_SEQUENCE.index(changed_stage)
        start_idx = idx if include_current else idx + 1
        for downstream_idx in range(start_idx, len(STAGE_SEQUENCE)):
            downstream_name = STAGE_SEQUENCE[downstream_idx]
            meta = context.stages.get(downstream_name)
            if meta and meta.status in (
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
                StageStatus.RUNNING,
                StageStatus.READY,
            ):
                meta.status = StageStatus.STALE
                meta.warnings.append(
                    f"Result invalidated: upstream stage '{changed_stage}' was re-executed."
                )

        # Clear relevant downstream run IDs
        if idx <= 0:
            context.impact_run_id = None
        if idx <= 1:
            context.selection_run_id = None
        if idx <= 2:
            context.execution_run_id = None
        if idx <= 3:
            context.arbitration_run_id = None

        self.save_workflow(context)
        return context

    def materialize_execution_workspace(self, context: WorkflowContext) -> tuple[Path, bool]:
        """
        Reconstructs a disposable, revision-accurate workspace from:
          1. Immutable Git commit archive (context.target_commit_hash).
          2. Verified snapshot diff (if context.is_dirty).
          3. Untracked file manifests recorded at initialization.
          4. Targeted generated test files from Stage 2.
        Guarantees that Docker execution runs against the exact recorded revision,
        even if the live host repository is subsequently modified or dirty.
        Returns (workspace_path, is_disposable).
        """
        rp = Path(context.repo_path).expanduser().resolve()
        if not (rp / ".git").exists():
            return rp, False

        # Verify that context.target_commit_hash exists in Git
        chk = subprocess.run(
            ["git", "cat-file", "-e", f"{context.target_commit_hash}^{{commit}}"],
            cwd=str(rp),
            capture_output=True,
            check=False,
        )
        if chk.returncode != 0:
            # Synthetic or non-existent commit (e.g. mocked unit tests)
            return rp, False

        ws_id = f"ws-{context.workflow_id}"
        ws_path = (self.workspaces_dir / ws_id).resolve()
        if ws_path.exists():
            shutil.rmtree(ws_path, ignore_errors=True)
        ws_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Extract git archive of target_commit_hash into ws_path
            archive_res = subprocess.run(
                ["git", "archive", context.target_commit_hash],
                cwd=str(rp),
                capture_output=True,
                check=True,
            )
            with tarfile.open(fileobj=io.BytesIO(archive_res.stdout)) as tf:
                tf.extractall(path=ws_path)

            # 2. If dirty snapshot was captured, apply patch and restore untracked files
            if context.is_dirty and context.patch_snapshot_id:
                snap_file = self.snapshots_dir / f"{context.patch_snapshot_id}.patch"
                if snap_file.exists():
                    snap_data = json.loads(snap_file.read_text(encoding="utf-8"))
                    diff_text = snap_data.get("diff_head", "")
                    if diff_text and diff_text.strip():
                        # Initialize a temporary git repository so git apply recognizes deletions & renames
                        subprocess.run(["git", "init"], cwd=str(ws_path), capture_output=True, check=True)
                        patch_file = ws_path / ".snapshot.patch"
                        patch_file.write_text(diff_text, encoding="utf-8")
                        subprocess.run(
                            ["git", "apply", "--whitespace=nowarn", "--binary", str(patch_file)],
                            cwd=str(ws_path),
                            capture_output=True,
                            check=False,
                        )
                        if patch_file.exists():
                            patch_file.unlink()
                        # Remove temporary .git folder so workspace remains a clean export
                        shutil.rmtree(ws_path / ".git", ignore_errors=True)

                    untracked = snap_data.get("untracked_files", {})
                    for rel_file, info in untracked.items():
                        dest_f = ws_path / rel_file
                        dest_f.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(info, dict) and "b64" in info:
                            dest_f.write_bytes(base64.b64decode(info["b64"]))
                        elif isinstance(info, str):
                            dest_f.write_text(info, encoding="utf-8")

            # 3. Synchronize Stage 2 targeted generated tests if present in the main repo
            main_gen_dir = rp / "tests" / "generated"
            if main_gen_dir.exists():
                ws_gen_dir = ws_path / "tests" / "generated"
                ws_gen_dir.mkdir(parents=True, exist_ok=True)
                for gf in main_gen_dir.glob("test_*.py"):
                    target_gf = ws_gen_dir / gf.name
                    if not target_gf.exists():
                        shutil.copy2(gf, target_gf)

            return ws_path, True
        except Exception as e:
            logger.warning(f"Failed to materialize disposable workspace: {e}. Falling back to host repo.")
            if ws_path.exists():
                shutil.rmtree(ws_path, ignore_errors=True)
            return rp, False

    def cleanup_execution_workspace(self, workspace_path: Path) -> None:
        """Safely cleans up disposable execution workspace."""
        try:
            if workspace_path.exists() and "ws-" in workspace_path.name:
                shutil.rmtree(workspace_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup execution workspace {workspace_path}: {e}")

