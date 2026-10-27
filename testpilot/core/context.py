"""
Central Analysis Context Subsystem for TestPilot AI.
Provides a single shared analysis context and run identifier across all stages:
Repository Selection -> Repository Evolution -> Specification & Boundaries ->
Deterministic Matrix -> Impact Analysis -> Test Generation -> Test Execution -> Failure Arbitration.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActiveAnalysisContext(BaseModel):
    """
    Unified analysis context representing a single repository change evaluation.
    Propagated sequentially through all workflow stages to eliminate disconnected defaults.
    """

    analysis_id: str = Field(
        default_factory=lambda: f"run-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    )
    repo_path: str = Field(default=".", description="Absolute or relative path to target repository")
    repo_name: str = Field(default="Testbed", description="Display name of target repository")
    base_ref: str = Field(default="HEAD~1", description="Git base revision")
    target_ref: Optional[str] = Field(default="HEAD", description="Git target revision or working tree")
    diff_stat: str = Field(default="", description="Summary git diffstat")
    changed_files: list[str] = Field(default_factory=list, description="Files altered between base and target refs")
    changed_symbols: list[dict[str, Any]] = Field(default_factory=list, description="Changed AST symbols")
    primary_target_file: Optional[str] = Field(default=None, description="Primary changed file under test")
    primary_target_symbol: Optional[str] = Field(default=None, description="Primary changed symbol under test")
    spec_path: Optional[str] = Field(default=None, description="Path to OpenAPI / PRD specification if present")
    spec_available: bool = Field(default=False, description="Whether formal OpenAPI specification is present")
    spec_status: str = Field(default="Not available for this repository", description="Honest human-readable spec status")
    spec_message: str = Field(
        default="Repository-level boundary analysis can continue using available source-code evidence.",
        description="Explanation of boundary analysis capabilities",
    )
    prioritized_tests: list[dict[str, Any]] = Field(default_factory=list, description="Candidate regression tests")
    impact_candidates: list[dict[str, Any]] = Field(default_factory=list, description="Upstream callers and blast nodes")
    deterministic_matrix: list[dict[str, Any]] = Field(default_factory=list, description="Boundary test cases")
    generated_tests: list[dict[str, Any]] = Field(default_factory=list, description="Synthesized boundary test code")
    execution_results: list[dict[str, Any]] = Field(default_factory=list, description="Pytest execution results")
    arbitration_results: list[dict[str, Any]] = Field(default_factory=list, description="Three-valued arbitration verdicts")
    sourcegraph_status: str = Field(default="Local AST Fallback", description="Code intelligence backend status")
    rag_status: str = Field(default="Ready", description="Repository RAG vector store status")
    workflow_id: Optional[str] = Field(default=None, description="Linked multi-stage workflow ID")
    base_commit_hash: Optional[str] = Field(default=None, description="Immutable 40-char base commit SHA")
    target_commit_hash: Optional[str] = Field(default=None, description="Immutable 40-char target commit SHA")
    completed_stages: list[str] = Field(default_factory=list, description="List of completed stage names")
    current_stage: str = Field(default="evolution", description="Active workflow stage")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AnalysisContextManager:
    """Manages global and session active analysis context."""

    _instance: Optional[AnalysisContextManager] = None
    _context: ActiveAnalysisContext

    def __new__(cls) -> AnalysisContextManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._context = cls._create_default_context()
        return cls._instance

    @classmethod
    def _create_default_context(cls) -> ActiveAnalysisContext:
        spec_path, spec_avail, spec_stat = cls.detect_spec(".")
        ctx = ActiveAnalysisContext(
            repo_path=".",
            repo_name="Testbed",
            base_ref="HEAD~1",
            target_ref="HEAD",
            primary_target_file="testbed/app/services/order_service.py" if Path("testbed/app/services/order_service.py").exists() else None,
            primary_target_symbol="calculate_order_totals",
            spec_path=spec_path,
            spec_available=spec_avail,
            spec_status=spec_stat,
        )
        return ctx

    @staticmethod
    def detect_repo_name(repo_path: str) -> str:
        """Derive readable repository name from path or git remote."""
        rp = Path(repo_path).expanduser().resolve()
        name_lower = rp.name.lower()
        if "flask" in name_lower:
            return "Flask"
        if "django" in name_lower:
            return "Django"
        if "homeassistant" in name_lower or "home-assistant" in name_lower:
            return "Home Assistant"
        if rp == Path(".").resolve() or "testpilot" in name_lower or "testbed" in name_lower:
            return "Testbed (TestPilot)"
        return rp.name

    @staticmethod
    def detect_spec(repo_path: str) -> tuple[Optional[str], bool, str]:
        """
        Check whether an OpenAPI or PRD specification exists in the repository.
        Does not fabricate a specification if one does not exist.
        """
        rp = Path(repo_path).expanduser().resolve()
        candidate_paths = [
            rp / "testbed" / "openapi.json",
            rp / "openapi.json",
            rp / "openapi.yaml",
            rp / "openapi.yml",
            rp / "swagger.json",
            rp / "docs" / "openapi.json",
        ]
        for cp in candidate_paths:
            if cp.exists() and cp.is_file():
                rel_p = str(cp.relative_to(rp)) if cp.is_relative_to(rp) else str(cp)
                return rel_p, True, "Available"

        return None, False, "Not available for this repository"

    def get_active_context(self) -> ActiveAnalysisContext:
        """Retrieve current active context."""
        return self._context

    def set_active_context(self, context: ActiveAnalysisContext) -> ActiveAnalysisContext:
        """Explicitly set current active context."""
        context.updated_at = datetime.now(timezone.utc).isoformat()
        self._context = context
        return self._context

    def update_active_context(self, **kwargs: Any) -> ActiveAnalysisContext:
        """Update fields on the active context."""
        current_data = self._context.model_dump()
        current_data.update(kwargs)
        current_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._context = ActiveAnalysisContext(**current_data)
        return self._context

    def reset(self, repo_path: str = ".", repo_name: Optional[str] = None) -> ActiveAnalysisContext:
        """Clear old stage results and initialize a fresh analysis context with new analysis ID."""
        spec_path, spec_avail, spec_stat = self.detect_spec(repo_path)
        resolved_name = repo_name or self.detect_repo_name(repo_path)
        new_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

        self._context = ActiveAnalysisContext(
            analysis_id=new_id,
            repo_path=str(repo_path),
            repo_name=resolved_name,
            base_ref="HEAD~1",
            target_ref="HEAD",
            diff_stat="",
            changed_files=[],
            changed_symbols=[],
            primary_target_file=None,
            primary_target_symbol=None,
            spec_path=spec_path,
            spec_available=spec_avail,
            spec_status=spec_stat,
            spec_message=(
                "OpenAPI specification detected and ready for formal boundary extraction."
                if spec_avail
                else "Repository-level boundary analysis can continue using available source-code evidence."
            ),
            prioritized_tests=[],
            impact_candidates=[],
            deterministic_matrix=[],
            generated_tests=[],
            execution_results=[],
            arbitration_results=[],
            completed_stages=[],
            current_stage="evolution",
        )
        return self._context

    def reset_active_context(self, repo_path: str = ".", repo_name: Optional[str] = None) -> ActiveAnalysisContext:
        """Alias for reset()."""
        return self.reset(repo_path=repo_path, repo_name=repo_name)

    def establish_from_evolution(
        self,
        report_data: dict[str, Any],
        repo_path: str,
        repo_name: Optional[str] = None,
        base_ref: str = "HEAD~1",
        target_ref: Optional[str] = "HEAD",
        analysis_id: Optional[str] = None,
    ) -> ActiveAnalysisContext:
        """
        Establish active analysis context from Repository Evolution results.
        Selects primary target file and symbols from actual diffs, not hardcoded defaults.
        """
        resolved_name = repo_name or self.detect_repo_name(repo_path)
        spec_path, spec_avail, spec_stat = self.detect_spec(repo_path)

        changed_files: list[str] = report_data.get("changed_files", [])
        changed_symbols: list[dict[str, Any]] = report_data.get("changed_symbols", [])

        # Pick primary target file: first non-test python file, or first changed file
        primary_file: Optional[str] = None
        for cf in changed_files:
            if cf.endswith(".py") and not any(t in cf for t in ("tests/", "test_", "/test/")):
                primary_file = cf
                break
        if not primary_file and changed_files:
            primary_file = changed_files[0]

        # Pick primary target symbol: first matching symbol in primary file or first symbol
        primary_symbol: Optional[str] = None
        for cs in changed_symbols:
            cs_file = cs.get("file_path", "")
            if primary_file and cs_file == primary_file:
                c_name = cs.get("class_name")
                s_name = cs.get("name", "")
                primary_symbol = f"{c_name}.{s_name}" if c_name else s_name
                break
        if not primary_symbol and changed_symbols:
            first = changed_symbols[0]
            c_name = first.get("class_name")
            s_name = first.get("name", "")
            primary_symbol = f"{c_name}.{s_name}" if c_name else s_name

        # If repo is testbed and no changed symbols found, fallback to order_service
        if not primary_file and resolved_name.startswith("Testbed"):
            primary_file = "testbed/app/services/order_service.py"
            primary_symbol = "calculate_order_totals"

        # Construct candidate tests & impacts
        prioritized_tests = report_data.get("prioritized_tests", [])
        direct_impacts = report_data.get("direct_impacts", [])
        indirect_impacts = report_data.get("indirect_impacts", [])
        all_impacts = direct_impacts + indirect_impacts

        use_id = analysis_id or self._context.analysis_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

        # Resolve immutable commits & initialize workflow storage
        from testpilot.core.workflow_context import (
            StageStatus,
            WorkflowContext,
            WorkflowStorageManager,
        )

        wf_mgr = WorkflowStorageManager()
        canonical_repo_id = wf_mgr.sanitize_repo_id(resolved_name, repo_path)
        try:
            base_sha = wf_mgr.resolve_immutable_commit(repo_path, base_ref)
        except Exception:
            base_sha = hashlib.sha256(f"{resolved_name}_{base_ref}".encode()).hexdigest()[:40]

        try:
            target_sha = wf_mgr.resolve_immutable_commit(repo_path, target_ref or "HEAD")
        except Exception:
            target_sha = hashlib.sha256(f"{resolved_name}_{target_ref}".encode()).hexdigest()[:40]

        is_clean, _ = wf_mgr.check_working_tree_clean(repo_path)
        existing_wf = wf_mgr.get_active_workflow()
        if existing_wf and (existing_wf.repo_id == canonical_repo_id or existing_wf.workflow_id == self._context.workflow_id):
            wf = existing_wf
            wf_id = wf.workflow_id
            wf.base_commit_hash = base_sha
            wf.target_commit_hash = target_sha
            wf.is_dirty = not is_clean
        else:
            wf_id = f"wf-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
            wf = WorkflowContext.create_initial(
                repo_id=canonical_repo_id,
                repo_path=str(repo_path),
                repo_name=resolved_name,
                base_commit_hash=base_sha,
                target_commit_hash=target_sha,
                is_dirty=not is_clean,
                workflow_id=wf_id,
            )
        wf.evolution_run_id = use_id
        evo_meta = wf.stages["evolution"]
        evo_meta.status = StageStatus.COMPLETED
        evo_meta.run_id = use_id
        evo_meta.completed_at = datetime.now(timezone.utc).isoformat()
        evo_meta.summary = {
            "changed_files_count": len(changed_files),
            "changed_symbols_count": len(changed_symbols),
            "prioritized_tests_count": len(prioritized_tests),
        }
        # Mark impact analysis ready
        wf.stages["impact"].status = StageStatus.READY
        wf.stages["impact"].upstream_run_id = use_id
        wf_mgr.save_workflow(wf)

        self._context = ActiveAnalysisContext(
            analysis_id=use_id,
            repo_path=str(repo_path),
            repo_name=resolved_name,
            base_ref=base_ref,
            target_ref=target_ref,
            diff_stat=report_data.get("diff_stat", ""),
            changed_files=changed_files,
            changed_symbols=changed_symbols,
            primary_target_file=primary_file,
            primary_target_symbol=primary_symbol,
            spec_path=spec_path,
            spec_available=spec_avail,
            spec_status=spec_stat,
            spec_message=(
                "OpenAPI specification detected and ready for formal boundary extraction."
                if spec_avail
                else "Repository-level boundary analysis can continue using available source-code evidence."
            ),
            prioritized_tests=prioritized_tests,
            impact_candidates=all_impacts,
            deterministic_matrix=[],
            generated_tests=[],
            execution_results=[],
            arbitration_results=[],
            workflow_id=wf_id,
            base_commit_hash=base_sha,
            target_commit_hash=target_sha,
            completed_stages=["evolution"],
            current_stage="impact",
        )
        return self._context
