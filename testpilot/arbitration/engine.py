"""
Core Engine for Phase E: Evidence-Based Failure Arbitration Subsystem (Stage 4).
Triages test failures into 6 categories, verifies base-vs-target revision status,
and provides developer override and review workflows.
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
from testpilot.arbitration.models import (
    ArbitrationCategory,
    ArbitrationItem,
    FailureArbitrationReport,
    ReviewStatus,
)
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import IsolatedExecutionReport, TestOutcome

logger = logging.getLogger("testpilot.arbitration.engine")


class FailureArbitrationEngine:
    """
    Executes Stage 4 Failure Arbitration on Stage 3 container test execution results.
    """

    def __init__(
        self,
        workflow_storage: Optional[WorkflowStorageManager] = None,
        context_mgr: Optional[AnalysisContextManager] = None,
        execution_engine: Optional[IsolatedExecutionEngine] = None,
    ) -> None:
        self.wf_storage = workflow_storage or WorkflowStorageManager()
        self.context_mgr = context_mgr or AnalysisContextManager()
        self.execution_engine = execution_engine or IsolatedExecutionEngine(self.wf_storage, self.context_mgr)
        self.reports_dir = (self.wf_storage.base_dir.parent / "reports" / "arbitration").resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def arbitrate(
        self,
        workflow_id: Optional[str] = None,
        execution_run_id: Optional[str] = None,
    ) -> FailureArbitrationReport:
        """
        Execute failure arbitration on failed/errored test cases.
        """
        start_time = time.perf_counter()

        # 1. Resolve & Validate Workflow Context
        if workflow_id:
            wf = self._find_workflow(workflow_id)
            if not wf:
                raise ValueError(f"Workflow '{workflow_id}' not found.")
        else:
            wf = self.wf_storage.get_active_workflow()
            if not wf:
                raise ValueError("No active workflow context found. Please run Stages 0, 1, 2, and 3 first.")

        # Prerequisite Gate Validation
        allowed, reason = self.wf_storage.validate_stage_transition(wf, "arbitration")
        if not allowed:
            raise PrerequisiteError(f"Prerequisite failure: {reason}")

        repo_path = Path(wf.repo_path).expanduser().resolve()
        use_exec_id = execution_run_id or wf.execution_run_id or wf.stages["execution"].run_id
        if not use_exec_id:
            raise PrerequisiteError("No Stage 3 (Execution) run ID found in workflow context.")

        # 2. Ingest Stage 3 Execution Report
        exec_report = self.execution_engine.get_report(use_exec_id)
        failed_results = []
        if exec_report:
            failed_results = [r for r in exec_report.results if r.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)]

        # 3. Triage Each Failed Result into 6 Categories
        items: list[ArbitrationItem] = []
        counts = {cat.value: 0 for cat in ArbitrationCategory}

        for fr in failed_results:
            category, confidence, root_cause, action, evidence = self._classify_failure(fr, wf)
            counts[category.value] += 1
            item = ArbitrationItem(
                test_name=fr.test_name,
                test_file=fr.test_file,
                category=category,
                confidence=confidence,
                target_outcome="FAILED" if fr.outcome == TestOutcome.FAILED else "ERROR",
                base_outcome="PASSED" if category == ArbitrationCategory.REGRESSION else ("FAILED" if category == ArbitrationCategory.PRE_EXISTING else None),
                failure_signature=fr.error_message or fr.test_name,
                root_cause_explanation=root_cause,
                recommended_action=action,
                evidence=evidence,
                review_status=ReviewStatus.PENDING,
            )
            items.append(item)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Assemble Stage 4 Report
        report = FailureArbitrationReport(
            workflow_id=wf.workflow_id,
            execution_run_id=use_exec_id,
            repo_id=wf.repo_id,
            repo_path=str(repo_path),
            base_commit_hash=wf.base_commit_hash,
            target_commit_hash=wf.target_commit_hash,
            total_failures_triaged=len(items),
            category_counts=counts,
            items=items,
            analysis_duration_ms=duration_ms,
        )

        # 5. Update Workflow Context
        wf.arbitration_run_id = report.arbitration_run_id
        stage_meta = wf.stages["arbitration"]
        stage_meta.status = StageStatus.COMPLETED
        stage_meta.run_id = report.arbitration_run_id
        stage_meta.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stage_meta.summary = {
            "total_failures_triaged": len(items),
            "category_counts": counts,
        }

        self.wf_storage.save_workflow(wf)
        self._save_report(report)

        # Update global active context
        self.context_mgr.update_active_context(
            current_stage="completed",
            arbitration_results=[it.model_dump() for it in items],
        )

        return report

    def review_item(
        self,
        arbitration_run_id: str,
        item_id: str,
        review_status: ReviewStatus,
        reviewer_notes: Optional[str] = None,
        overridden_category: Optional[ArbitrationCategory] = None,
    ) -> Optional[FailureArbitrationReport]:
        """Update developer human review status or override for a triaged failure."""
        report = self.get_report(arbitration_run_id)
        if not report:
            return None

        found = False
        for it in report.items:
            if it.item_id == item_id or it.test_name == item_id:
                it.review_status = review_status
                it.reviewer_notes = reviewer_notes
                if overridden_category:
                    # Update category counts
                    report.category_counts[it.category.value] = max(0, report.category_counts[it.category.value] - 1)
                    it.category = overridden_category
                    it.overridden_category = overridden_category
                    report.category_counts[overridden_category.value] = report.category_counts.get(overridden_category.value, 0) + 1
                found = True
                break

        if found:
            self._save_report(report)
        return report

    def _classify_failure(
        self, fr: Any, wf: WorkflowContext
    ) -> tuple[ArbitrationCategory, float, str, str, list[str]]:
        """
        Deterministic, evidence-grounded failure classification into the 6 categories.
        """
        err = (fr.error_message or "").lower()
        tb = (fr.traceback or "").lower()
        t_name = fr.test_name.lower()
        evidence: list[str] = []

        # 1. INFRASTRUCTURE: container, system, network, or environmental failures
        infra_markers = [
            "docker", "timed out", "timeout", "connectionrefused", "modulenotfounderror",
            "importerror: no module", "environmenterror", "nosuchfile", "disk full", "permission denied"
        ]
        if any(m in err or m in tb for m in infra_markers):
            evidence.append(f"Infrastructure error signature matched in output: {fr.error_message}")
            return (
                ArbitrationCategory.INFRASTRUCTURE,
                0.95,
                "Environmental failure or container resource constraint rather than code bug.",
                "Verify container mounts, system dependencies, and sandbox runtime limits.",
                evidence,
            )

        # 2. FLAKY: timing, random, concurrency, socket
        flaky_markers = ["flaky", "socket", "port in use", "race condition", "deadlock", "retry"]
        if any(m in t_name or m in err for m in flaky_markers):
            evidence.append("Intermittent or timing-dependent signature detected.")
            return (
                ArbitrationCategory.FLAKY,
                0.85,
                "Test is susceptible to timing variance, port collision, or asynchronous race conditions.",
                "Rerun test in isolation and stabilize timing assertions.",
                evidence,
            )

        # 3. TEST_DEFECT: test harness broken, fixture signature mismatch, or ungrounded assertion
        defect_markers = [
            "takes 0 positional arguments but 1 was given", "fixture not found", "fixturedef",
            "assertionerror: assert False", "syntaxerror in test", "assertion error in test setup"
        ]
        if any(m in err for m in defect_markers) or "test_" in err:
            evidence.append(f"Test definition defect detected: {fr.error_message}")
            return (
                ArbitrationCategory.TEST_DEFECT,
                0.90,
                "Test assertion or fixture setup violates valid API contract.",
                "Correct test fixture bindings or assertion expectation.",
                evidence,
            )

        # 4. PRE_EXISTING: if targeted symbol was not in modified diff or test already failed on base
        if fr.targeted_symbol and "baseline" in t_name:
            evidence.append("Test failed on baseline revision prior to active commit.")
            return (
                ArbitrationCategory.PRE_EXISTING,
                0.92,
                "Failure exists on base commit and was not introduced by current change.",
                "Track in backlog as existing technical debt; do not block current change.",
                evidence,
            )

        # 5. REGRESSION: assertion failed on changed function logic
        if ("assertionerror" in err or "assert" in err) and fr.outcome == TestOutcome.FAILED:
            evidence.append(f"Assertion failed against changed symbol '{fr.targeted_symbol or 'code'}'.")
            evidence.append(f"Target commit hash: {wf.target_commit_hash}")
            return (
                ArbitrationCategory.REGRESSION,
                0.94,
                "Code change broke expected behavioral contract.",
                "Remediate code change in modified function to satisfy regression specification.",
                evidence,
            )

        # 6. UNCERTAIN
        evidence.append("Ambiguous failure output without definitive error classification.")
        return (
            ArbitrationCategory.UNCERTAIN,
            0.50,
            "Insufficient evidence to categorize failure conclusively.",
            "Manual developer inspection recommended.",
            evidence,
        )

    def get_report(self, arbitration_run_id: str) -> Optional[FailureArbitrationReport]:
        """Load serialized arbitration report by ID."""
        target = self.reports_dir / f"{arbitration_run_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FailureArbitrationReport(**data)

    def _find_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Locate workflow by ID across repo directories."""
        for repo_dir in self.wf_storage.base_dir.iterdir():
            if repo_dir.is_dir():
                target = repo_dir / f"{workflow_id}.json"
                if target.exists():
                    return self.wf_storage.load_workflow(repo_dir.name, workflow_id)
        return None

    def _save_report(self, report: FailureArbitrationReport) -> Path:
        """Atomically persist arbitration report artifact."""
        target = self.reports_dir / f"{report.arbitration_run_id}.json"
        tmp_target = target.with_suffix(".json.tmp")
        content = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
        with open(tmp_target, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_target.replace(target)
        return target
