"""
Unit and integration tests for Phase C: Intelligent Test Selection & Targeted Generation.
Verifies:
1. Prerequisite enforcement (Stage 1 Impact Analysis must be COMPLETED).
2. Selection and ranking of existing regression tests with explainable criteria.
3. Suite reduction and execution time efficiency calculations.
4. 5-Criterion test qualification standard for synthesized targeted tests.
5. Strict isolation between generated tests and benchmark ground truth.
6. Workflow state transition (Stage 2 COMPLETED, Stage 3 READY).
7. REST API endpoints (/api/selection/select, /api/selection/generate, /api/selection/{id}).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageStatus,
    PrerequisiteError,
)
from testpilot.impact.engine import ImpactAnalysisEngine
from testpilot.selection.engine import TestSelectionEngine
from testpilot.web.api import app

client = TestClient(app)


def test_selection_prerequisite_enforcement(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = TestSelectionEngine(workflow_storage=mgr)

    # Workflow where Impact is NOT_STARTED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-sel-prereq",
    )
    # Mark evolution done but impact not done
    wf.stages["evolution"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    with pytest.raises(PrerequisiteError) as exc_info:
        engine.select(workflow_id="wf-test-sel-prereq")
    assert "impact" in str(exc_info.value).lower()


def test_selection_execution_and_efficiency(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    impact_engine = ImpactAnalysisEngine(workflow_storage=mgr)
    selection_engine = TestSelectionEngine(workflow_storage=mgr, impact_engine=impact_engine)

    # Setup workflow where Evolution is COMPLETED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-sel-run",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["evolution"].run_id = "evo-run-002"
    mgr.save_workflow(wf)

    # Run Stage 1 (Impact Analysis) first
    impact_report = impact_engine.analyze(workflow_id="wf-test-sel-run", max_depth=2)
    assert impact_report.impact_run_id.startswith("imp-")

    # Run Stage 2 (Intelligent Test Selection)
    sel_report = selection_engine.select(
        workflow_id="wf-test-sel-run",
        impact_run_id=impact_report.impact_run_id,
        generate_targeted=True,
        max_targeted_tests=3,
    )

    assert sel_report.selection_run_id.startswith("sel-")
    assert sel_report.workflow_id == "wf-test-sel-run"
    assert sel_report.impact_run_id == impact_report.impact_run_id
    assert len(sel_report.selected_tests) > 0

    # Efficiency metrics check
    eff = sel_report.efficiency
    assert eff.total_repository_tests > 0
    assert eff.selected_existing_tests > 0
    assert 0.0 <= eff.suite_reduction_percentage <= 100.0
    assert eff.time_savings_percentage >= 0.0

    # 5-Criterion validation check for generated tests
    assert len(sel_report.targeted_generated_tests) > 0
    for tgt in sel_report.targeted_generated_tests:
        assert tgt.qualification_passed is True
        details = tgt.qualification_details
        assert details["ast_syntax_valid"] is True
        assert details["collectable"] is True
        assert details["has_non_trivial_assertions"] is True
        assert details["contract_grounded"] is True
        assert details["isolated_from_benchmark"] is True

    # State transition check
    updated_wf = mgr.load_workflow(wf.repo_id, wf.workflow_id)
    assert updated_wf.selection_run_id == sel_report.selection_run_id
    assert updated_wf.stages["selection"].status == StageStatus.COMPLETED
    assert updated_wf.stages["execution"].status == StageStatus.READY
    assert updated_wf.stages["execution"].upstream_run_id == sel_report.selection_run_id


def test_selection_api_endpoints():
    """Verify REST API endpoints: /api/selection/select, /api/selection/generate, /api/selection/{id}."""
    # 1. Initialize workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]

    # 2. Complete Evolution and Impact stages
    from testpilot.web.api import workflow_storage, impact_engine
    wf = workflow_storage.load_workflow(wf_data["repo_id"], wf_id)
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["evolution"].run_id = "evo-sel-api-test"
    workflow_storage.save_workflow(wf)

    imp_report = impact_engine.analyze(workflow_id=wf_id, max_depth=2)

    # 3. Call POST /api/selection/select
    sel_res = client.post(
        "/api/selection/select",
        json={"workflow_id": wf_id, "impact_run_id": imp_report.impact_run_id, "max_targeted_tests": 2},
    )
    assert sel_res.status_code == 200
    data = sel_res.json()
    assert data["selection_run_id"].startswith("sel-")
    assert "efficiency" in data
    assert "selected_tests" in data
    sel_id = data["selection_run_id"]

    # 4. Call GET /api/selection/{selection_run_id}
    get_res = client.get(f"/api/selection/{sel_id}")
    assert get_res.status_code == 200
    assert get_res.json()["selection_run_id"] == sel_id

    # 5. Call POST /api/selection/generate
    gen_res = client.post(
        "/api/selection/generate",
        json={"workflow_id": wf_id, "impact_run_id": imp_report.impact_run_id, "max_tests": 2},
    )
    assert gen_res.status_code == 200
    assert "targeted_tests" in gen_res.json()
