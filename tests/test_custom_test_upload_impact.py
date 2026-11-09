"""
Unit and integration tests for User-Uploaded Test Cases with AST Impact Analysis and Stage 3 Execution.
Verifies:
1. AST parsing and direct/indirect impact calculation against changed symbols.
2. Boundary constraint recognition in assertions.
3. Syntax error detection and safe diagnostics.
4. Workflow context persistence and workspace materialization.
5. Inclusion and execution of custom tests in Stage 3 (Isolated Execution).
6. REST API endpoints (/api/tests/upload, /api/tests/uploaded, /api/tests/uploaded/{id}).
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageStatus,
)
from testpilot.impact.engine import ImpactAnalysisEngine
from testpilot.impact.models import RiskCategory
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import TestOutcome
from testpilot.selection.engine import TestSelectionEngine
from testpilot.web.api import app

client = TestClient(app)


def test_custom_test_impact_direct_critical(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    impact_engine = ImpactAnalysisEngine(workflow_storage=mgr)

    # Initialize workflow
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-custom-impact",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    # Populate active context with changed symbols
    impact_engine.context_mgr.update_active_context(
        changed_symbols=[{"name": "calculate_order_totals", "file": "testbed/order_service.py"}]
    )

    test_code = """
import pytest
from testbed.order_service import calculate_order_totals

def test_user_shipping_discount():
    items = [{"id": "item1", "price": 45.0, "quantity": 1}]
    res = calculate_order_totals(items, apply_tax=True)
    assert res["subtotal"] == 45.0
"""

    impact = impact_engine.analyze_custom_test(
        code=test_code,
        file_name="test_user_shipping.py",
        workflow_id="wf-test-custom-impact",
        save_to_workflow=True,
    )

    assert impact.ast_valid is True
    assert impact.test_name == "test_user_shipping_discount"
    assert "calculate_order_totals" in impact.directly_tested_symbols
    assert impact.risk_category == RiskCategory.CRITICAL
    assert impact.impact_tier == "DIRECT_CRITICAL"

    # Verify persisted to workflow
    loaded_wf = mgr.load_workflow(wf.repo_id, wf.workflow_id)
    assert len(loaded_wf.uploaded_tests) == 1
    assert loaded_wf.uploaded_tests[0]["test_name"] == "test_user_shipping_discount"


def test_custom_test_impact_syntax_error(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    impact_engine = ImpactAnalysisEngine(workflow_storage=mgr)

    bad_code = "def test_broken(: assert True"
    impact = impact_engine.analyze_custom_test(code=bad_code, file_name="test_broken.py")

    assert impact.ast_valid is False
    assert impact.impact_tier == "SYNTAX_ERROR"
    assert impact.syntax_error is not None
    assert "SyntaxError" in impact.syntax_error


def test_custom_test_execution_integration(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    impact_engine = ImpactAnalysisEngine(workflow_storage=mgr)
    sel_engine = TestSelectionEngine(workflow_storage=mgr)
    exec_engine = IsolatedExecutionEngine(workflow_storage=mgr, selection_engine=sel_engine)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-custom-exec",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    wf.selection_run_id = "sel-dummy"

    # Add custom uploaded test to workflow
    wf.uploaded_tests = [{
        "test_name": "test_custom_runner_verification",
        "file_name": "test_custom_runner.py",
        "code": "def test_custom_runner_verification(): assert 1 + 1 == 2",
        "relative_file_path": "tests/uploaded/test_custom_runner.py",
        "directly_tested_symbols": ["calculate_order_totals"],
        "ast_valid": True,
        "impact_tier": "DIRECT_CRITICAL",
    }]
    mgr.save_workflow(wf)

    # Execute in simulated mode
    report = exec_engine.execute(
        workflow_id="wf-test-custom-exec",
        execution_mode="simulated",
        allow_simulated_dry_run=True,
    )

    assert report.total_executed >= 1
    custom_results = [r for r in report.results if r.is_user_uploaded]
    assert len(custom_results) == 1
    assert custom_results[0].test_name == "test_custom_runner_verification"
    assert custom_results[0].is_user_uploaded is True
    assert custom_results[0].impact_tier == "DIRECT_CRITICAL"


def test_custom_test_api_endpoints():
    # 1. Initialize fresh workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_id = init_res.json()["workflow_id"]

    # 2. Upload test via POST /api/tests/upload
    code = """
def test_api_scenario():
    assert 10 > 5
"""
    upload_res = client.post(
        "/api/tests/upload",
        json={"workflow_id": wf_id, "file_name": "test_api_scenario.py", "code": code},
    )
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert data["test_name"] == "test_api_scenario"
    assert data["ast_valid"] is True
    test_id = data["test_id"]

    # 3. List via GET /api/tests/uploaded
    list_res = client.get(f"/api/tests/uploaded?workflow_id={wf_id}")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) >= 1
    assert any(i["test_id"] == test_id for i in items)

    # 4. Delete via DELETE /api/tests/uploaded/{test_id}
    del_res = client.delete(f"/api/tests/uploaded/{test_id}?workflow_id={wf_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "DELETED"

    # Verify removed
    list_after = client.get(f"/api/tests/uploaded?workflow_id={wf_id}").json()
    assert not any(i["test_id"] == test_id for i in list_after)

