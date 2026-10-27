"""
Core Execution Engine for Phase D: Isolated Test Execution with Docker.
Coordinates container sandboxes, enforces stage prerequisites, and feeds results to Stage 4 Failure Arbitration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Optional

from testpilot.core.context import AnalysisContextManager
from testpilot.core.workflow_context import (
    PrerequisiteError,
    StageStatus,
    WorkflowContext,
    WorkflowStorageManager,
)
from testpilot.execution.models import (
    DockerDiagnostics,
    DockerSecurityProfile,
    IsolatedExecutionReport,
    TestCaseResult,
    TestOutcome,
)
from testpilot.execution.runner import (
    DockerUnavailableError,
    PytestDockerRunner,
    TestRunner,
    UnsupportedFrameworkError,
)
from testpilot.selection.engine import TestSelectionEngine
from testpilot.selection.models import TestSelectionReport

logger = logging.getLogger("testpilot.execution.engine")


class IsolatedExecutionEngine:
    """
    Coordinates disposable container execution of Stage 2 selected and generated tests.
    """

    def __init__(
        self,
        workflow_storage: Optional[WorkflowStorageManager] = None,
        context_mgr: Optional[AnalysisContextManager] = None,
        selection_engine: Optional[TestSelectionEngine] = None,
        runner: Optional[TestRunner] = None,
    ) -> None:
        self.wf_storage = workflow_storage or WorkflowStorageManager()
        self.context_mgr = context_mgr or AnalysisContextManager()
        self.selection_engine = selection_engine or TestSelectionEngine(self.wf_storage, self.context_mgr)
        self.runner = runner or PytestDockerRunner()
        self.reports_dir = (self.wf_storage.base_dir.parent / "reports" / "execution").resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_docker_status(self) -> DockerDiagnostics:
        """Inspect Docker daemon health and return actionable diagnostics."""
        return self.runner.check_docker_availability()

    def execute(
        self,
        workflow_id: Optional[str] = None,
        selection_run_id: Optional[str] = None,
        allow_simulated_dry_run: bool = False,
        framework_name: str = "pytest",
    ) -> IsolatedExecutionReport:
        """
        Execute containerized tests for the specified workflow.
        """
        start_time = time.perf_counter()

        # 1. Framework validation
        if not self.runner.supports_framework(framework_name):
            raise UnsupportedFrameworkError(
                f"Unsupported test framework '{framework_name}'. Supported frameworks: pytest, unittest."
            )

        # 2. Resolve & Validate Workflow Context
        if workflow_id:
            wf = self._find_workflow(workflow_id)
            if not wf:
                raise ValueError(f"Workflow '{workflow_id}' not found.")
        else:
            wf = self.wf_storage.get_active_workflow()
            if not wf:
                raise ValueError("No active workflow context found. Please run Stage 0, 1, and 2 first.")

        # Prerequisite Gate Validation
        allowed, reason = self.wf_storage.validate_stage_transition(wf, "execution")
        if not allowed:
            raise PrerequisiteError(f"Prerequisite failure: {reason}")

        repo_path = Path(wf.repo_path).expanduser().resolve()
        use_sel_id = selection_run_id or wf.selection_run_id or wf.stages["selection"].run_id
        if not use_sel_id:
            raise PrerequisiteError("No Stage 2 (Selection) run ID found in workflow context.")

        # 3. Ingest Stage 2 Selection Report
        sel_report = self.selection_engine.get_report(use_sel_id)
        test_cases_to_run: list[dict[str, Any]] = []

        if sel_report:
            for st in sel_report.selected_tests:
                test_cases_to_run.append(st.model_dump())
            for tgt in sel_report.targeted_generated_tests:
                if tgt.qualification_passed:
                    test_cases_to_run.append({
                        "test_name": tgt.test_name,
                        "test_file": tgt.file_path,
                        "targeted_symbol": tgt.target_symbol,
                    })

        # Fallback to testbed if empty
        if not test_cases_to_run:
            active_ctx = self.context_mgr.get_active_context()
            for pt in active_ctx.prioritized_tests[:5]:
                test_cases_to_run.append(pt)

        if not test_cases_to_run:
            test_cases_to_run = [{
                "test_name": "test_subtotal_below_free_shipping_threshold",
                "test_file": "tests/generated/test_order_service.py",
                "targeted_symbol": "calculate_order_totals",
            }]

        # 4. Materialize disposable, revision-accurate execution workspace
        exec_ws, is_disposable = self.wf_storage.materialize_execution_workspace(wf)
        try:
            sec = DockerSecurityProfile()
            results, diagnostics = self.runner.run_tests(
                test_cases=test_cases_to_run,
                repo_path=exec_ws,
                security_profile=sec,
                allow_simulated_dry_run=allow_simulated_dry_run,
            )
        finally:
            if is_disposable:
                self.wf_storage.cleanup_execution_workspace(exec_ws)

        passed_count = sum(1 for r in results if r.outcome == TestOutcome.PASSED)
        failed_count = sum(1 for r in results if r.outcome == TestOutcome.FAILED)
        error_count = sum(1 for r in results if r.outcome == TestOutcome.ERROR)
        skipped_count = sum(1 for r in results if r.outcome == TestOutcome.SKIPPED)
        failed_names = [r.test_name for r in results if r.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)]

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Assemble Stage 3 Report
        report = IsolatedExecutionReport(
            workflow_id=wf.workflow_id,
            selection_run_id=use_sel_id,
            repo_id=wf.repo_id,
            repo_path=str(repo_path),
            base_commit_hash=wf.base_commit_hash,
            target_commit_hash=wf.target_commit_hash,
            total_executed=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            skipped_count=skipped_count,
            results=results,
            failed_test_names=failed_names,
            is_docker_isolated=diagnostics.is_available,
            security_profile=sec,
            docker_diagnostics=diagnostics,
            execution_duration_ms=duration_ms,
        )

        # 6. Update Workflow Context
        wf.execution_run_id = report.execution_run_id
        stage_meta = wf.stages["execution"]
        stage_meta.status = StageStatus.COMPLETED if (failed_count + error_count) == 0 else StageStatus.COMPLETED_WITH_WARNINGS
        stage_meta.run_id = report.execution_run_id
        stage_meta.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stage_meta.summary = {
            "total_executed": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "is_docker_isolated": diagnostics.is_available,
        }

        # Enable Stage 4 (Arbitration)
        wf.stages["arbitration"].status = StageStatus.READY
        wf.stages["arbitration"].upstream_run_id = report.execution_run_id

        self.wf_storage.save_workflow(wf)
        self._save_report(report)

        # Update global active context
        self.context_mgr.update_active_context(
            current_stage="arbitration",
            execution_results=[r.model_dump() for r in results],
        )

        return report

    def get_report(self, execution_run_id: str) -> Optional[IsolatedExecutionReport]:
        """Load serialized execution report by ID."""
        target = self.reports_dir / f"{execution_run_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IsolatedExecutionReport(**data)

    def _find_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Locate workflow by ID across repo directories."""
        for repo_dir in self.wf_storage.base_dir.iterdir():
            if repo_dir.is_dir():
                target = repo_dir / f"{workflow_id}.json"
                if target.exists():
                    return self.wf_storage.load_workflow(repo_dir.name, workflow_id)
        return None

    def _save_report(self, report: IsolatedExecutionReport) -> Path:
        """Atomically persist execution report artifact."""
        target = self.reports_dir / f"{report.execution_run_id}.json"
        tmp_target = target.with_suffix(".json.tmp")
        content = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
        with open(tmp_target, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_target.replace(target)
        return target
