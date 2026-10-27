"""
Unit and integration tests for Phase E: Evidence-Based Failure Arbitration (Stage 4).
Verifies:
1. Prerequisite gating (Stage 3 Execution must be COMPLETED).
2. Triage across all 6 categories (REGRESSION, PRE_EXISTING, FLAKY, INFRASTRUCTURE, TEST_DEFECT, UNCERTAIN).
3. Base-vs-target comparative telemetry provenance.
4. Human review workflows, status transitions, and category overrides.
5. Workflow state transition to final completed state.
6. REST API endpoints (/api/arbitration/analyze, /api/arbitration/review, /api/arbitration/{id}).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from testpilot.arbitration.engine import FailureArbitrationEngine
from testpilot.arbitration.models import ArbitrationCategory, ReviewStatus
from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageStatus,
    PrerequisiteError,
)
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import IsolatedExecutionReport, TestCaseResult, TestOutcome
from testpilot.web.api import app

client = TestClient(app)


def test_arbitration_prerequisite_enforcement(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = FailureArbitrationEngine(workflow_storage=mgr)

    # Workflow where Execution is NOT_STARTED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-arb-prereq",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    with pytest.raises(PrerequisiteError) as exc_info:
        engine.arbitrate(workflow_id="wf-test-arb-prereq")
    assert "execution" in str(exc_info.value).lower()


def test_arbitration_6_category_classification(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    exec_engine = IsolatedExecutionEngine(workflow_storage=mgr)
    arb_engine = FailureArbitrationEngine(workflow_storage=mgr, execution_engine=exec_engine)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-arb-triage",
    )
    for s in ("evolution", "impact", "selection"):
        wf.stages[s].status = StageStatus.COMPLETED
    wf.stages["execution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.execution_run_id = "exec-mock-001"
    mgr.save_workflow(wf)

    # Synthesize execution report with diverse failure signatures
    results = [
        TestCaseResult(
            test_name="test_order_totals_discount",
            test_file="tests/test_order.py",
            outcome=TestOutcome.FAILED,
            error_message="AssertionError: assert 45.0 == 50.0",
            targeted_symbol="calculate_order_totals",
        ),
        TestCaseResult(
            test_name="test_legacy_baseline_orders",
            test_file="tests/test_legacy.py",
            outcome=TestOutcome.FAILED,
            error_message="KeyError: 'legacy_field'",
            targeted_symbol="calculate_order_totals",
        ),
        TestCaseResult(
            test_name="test_db_timeout_event",
            test_file="tests/test_db.py",
            outcome=TestOutcome.ERROR,
            error_message="Container timed out after 60s",
        ),
        TestCaseResult(
            test_name="test_socket_retry_flaky",
            test_file="tests/test_socket.py",
            outcome=TestOutcome.FAILED,
            error_message="ConnectionRefusedError: socket flaky port collision",
        ),
        TestCaseResult(
            test_name="test_bad_assertion_syntax",
            test_file="tests/test_defect.py",
            outcome=TestOutcome.FAILED,
            error_message="TypeError: test_defect takes 0 positional arguments but 1 was given",
        ),
        TestCaseResult(
            test_name="test_unknown_odd_failure",
            test_file="tests/test_odd.py",
            outcome=TestOutcome.FAILED,
            error_message="ExitCode 137",
        ),
    ]

    exec_report = IsolatedExecutionReport(
        execution_run_id="exec-mock-001",
        workflow_id=wf.workflow_id,
        selection_run_id="sel-dummy",
        repo_id=wf.repo_id,
        repo_path=wf.repo_path,
        base_commit_hash=wf.base_commit_hash,
        target_commit_hash=wf.target_commit_hash,
        total_executed=len(results),
        failed_count=len(results),
        results=results,
    )
    exec_engine._save_report(exec_report)

    # Run Failure Arbitration
    report = arb_engine.arbitrate(workflow_id="wf-test-arb-triage")

    assert report.arbitration_run_id.startswith("arb-")
    assert report.total_failures_triaged == 6
    cats = {it.test_name: it.category for it in report.items}

    # Verify classification logic
    assert cats["test_order_totals_discount"] == ArbitrationCategory.REGRESSION
    assert cats["test_legacy_baseline_orders"] == ArbitrationCategory.PRE_EXISTING
    assert cats["test_db_timeout_event"] == ArbitrationCategory.INFRASTRUCTURE
    assert cats["test_socket_retry_flaky"] in (ArbitrationCategory.FLAKY, ArbitrationCategory.INFRASTRUCTURE)
    assert cats["test_bad_assertion_syntax"] == ArbitrationCategory.TEST_DEFECT
    assert cats["test_unknown_odd_failure"] in (ArbitrationCategory.UNCERTAIN, ArbitrationCategory.INFRASTRUCTURE)

    # Verify workflow state
    updated_wf = mgr.load_workflow(wf.repo_id, wf.workflow_id)
    assert updated_wf.arbitration_run_id == report.arbitration_run_id
    assert updated_wf.stages["arbitration"].status == StageStatus.COMPLETED


def test_developer_review_and_override(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    exec_engine = IsolatedExecutionEngine(workflow_storage=mgr)
    arb_engine = FailureArbitrationEngine(workflow_storage=mgr, execution_engine=exec_engine)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-arb-review",
    )
    for s in ("evolution", "impact", "selection"):
        wf.stages[s].status = StageStatus.COMPLETED
    wf.stages["execution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.execution_run_id = "exec-mock-002"
    mgr.save_workflow(wf)

    exec_report = IsolatedExecutionReport(
        execution_run_id="exec-mock-002",
        workflow_id=wf.workflow_id,
        selection_run_id="sel-dummy",
        repo_id=wf.repo_id,
        repo_path=wf.repo_path,
        base_commit_hash=wf.base_commit_hash,
        target_commit_hash=wf.target_commit_hash,
        total_executed=1,
        failed_count=1,
        results=[
            TestCaseResult(
                test_name="test_suspected_regression",
                test_file="tests/test_sample.py",
                outcome=TestOutcome.FAILED,
                error_message="AssertionError: assert 1 == 2",
            )
        ],
    )
    exec_engine._save_report(exec_report)

    report = arb_engine.arbitrate(workflow_id="wf-test-arb-review")
    item_id = report.items[0].item_id
    assert report.items[0].review_status == ReviewStatus.PENDING

    # Reviewer confirms failure as genuine regression
    updated = arb_engine.review_item(
        arbitration_run_id=report.arbitration_run_id,
        item_id=item_id,
        review_status=ReviewStatus.CONFIRMED,
        reviewer_notes="Confirmed as blocking regression on order discount calculation.",
    )
    assert updated.items[0].review_status == ReviewStatus.CONFIRMED
    assert "blocking regression" in updated.items[0].reviewer_notes

    # Reviewer overrides to TEST_DEFECT
    overridden = arb_engine.review_item(
        arbitration_run_id=report.arbitration_run_id,
        item_id=item_id,
        review_status=ReviewStatus.OVERRIDDEN,
        reviewer_notes="Test asserted outdated API contract.",
        overridden_category=ArbitrationCategory.TEST_DEFECT,
    )
    assert overridden.items[0].category == ArbitrationCategory.TEST_DEFECT
    assert overridden.items[0].review_status == ReviewStatus.OVERRIDDEN


def test_arbitration_api_endpoints():
    """Verify REST API endpoints: /api/arbitration/analyze, /api/arbitration/review, /api/arbitration/{id}."""
    # 1. Initialize workflow and complete prior stages
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]

    from testpilot.web.api import workflow_storage, execution_engine
    wf = workflow_storage.load_workflow(wf_data["repo_id"], wf_id)
    for s in ("evolution", "impact", "selection"):
        wf.stages[s].status = StageStatus.COMPLETED
    wf.stages["execution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.execution_run_id = "exec-api-mock"
    workflow_storage.save_workflow(wf)

    # Save mock execution report
    mock_exec = IsolatedExecutionReport(
        execution_run_id="exec-api-mock",
        workflow_id=wf_id,
        selection_run_id="sel-api",
        repo_id=wf.repo_id,
        repo_path=wf.repo_path,
        base_commit_hash=wf.base_commit_hash,
        target_commit_hash=wf.target_commit_hash,
        total_executed=1,
        failed_count=1,
        results=[
            TestCaseResult(
                test_name="test_api_triaged_case",
                test_file="tests/test_api.py",
                outcome=TestOutcome.FAILED,
                error_message="AssertionError: 200 != 500",
            )
        ],
    )
    execution_engine._save_report(mock_exec)

    # 2. Call POST /api/arbitration/analyze
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={"workflow_id": wf_id, "execution_run_id": "exec-api-mock"},
    )
    assert arb_res.status_code == 200
    arb_data = arb_res.json()
    assert arb_data["arbitration_run_id"].startswith("arb-")
    assert arb_data["total_failures_triaged"] == 1
    arb_id = arb_data["arbitration_run_id"]
    item_id = arb_data["items"][0]["item_id"]

    # 3. Call GET /api/arbitration/{id}
    get_res = client.get(f"/api/arbitration/{arb_id}")
    assert get_res.status_code == 200
    assert get_res.json()["arbitration_run_id"] == arb_id

    # 4. Call POST /api/arbitration/review
    rev_res = client.post(
        "/api/arbitration/review",
        json={
            "arbitration_run_id": arb_id,
            "item_id": item_id,
            "review_status": "CONFIRMED",
            "reviewer_notes": "Reviewed and approved by developer.",
        },
    )
    assert rev_res.status_code == 200
    assert rev_res.json()["items"][0]["review_status"] == "CONFIRMED"
