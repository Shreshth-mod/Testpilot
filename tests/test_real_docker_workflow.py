"""
Phase 5 & 7: Verification of Real Docker-Based Test Execution in TestPilot Workflow.
Enforces allow_simulated_dry_run=False, asserting genuine container execution.
"""

from fastapi.testclient import TestClient
import pytest

from testpilot.core.workflow_context import StageStatus, WorkflowContext
from testpilot.web.api import app, workflow_storage

client = TestClient(app)


def test_real_docker_isolated_workflow():
    """
    Executes the entire 5-stage sequential workflow requiring REAL Docker execution.
    Fails immediately if Docker is offline or if simulated fallback is used.
    """
    # 0. Confirm Docker is Available
    status_res = client.get("/api/execution/docker/status")
    assert status_res.status_code == 200
    docker_diag = status_res.json()
    assert docker_diag["is_available"] is True, f"Docker daemon must be online for real execution: {docker_diag.get('error_message')}"

    # 1. Workflow Initialization
    init_res = client.post(
        "/api/workflow/initialize",
        json={
            "repo_path": ".",
            "base_ref": "HEAD~1",
            "target_ref": "HEAD",
            "allow_dirty": True,
        },
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]
    repo_id = wf_data["repo_id"]

    # 2. Stage 0: Repository Evolution
    evo_res = client.post(
        "/api/evolution/analyze",
        json={"repo_path": ".", "workflow_id": wf_id},
    )
    assert evo_res.status_code == 200

    # 3. Stage 1: Impact Analysis
    imp_res = client.post(
        "/api/impact/analyze",
        json={"workflow_id": wf_id, "max_depth": 2, "force_refresh": True},
    )
    assert imp_res.status_code == 200
    imp_data = imp_res.json()
    imp_id = imp_data["impact_run_id"]

    # 4. Stage 2: Intelligent Test Selection & Generation
    sel_res = client.post(
        "/api/selection/select",
        json={
            "workflow_id": wf_id,
            "impact_run_id": imp_id,
            "generate_targeted": True,
            "max_targeted_tests": 2,
        },
    )
    assert sel_res.status_code == 200
    sel_data = sel_res.json()
    sel_id = sel_data["selection_run_id"]

    # 5. Stage 3: REAL Isolated Test Execution with Docker
    # CRITICAL: allow_simulated_dry_run MUST BE False!
    exec_res = client.post(
        "/api/execution/run",
        json={
            "workflow_id": wf_id,
            "selection_run_id": sel_id,
            "allow_simulated_dry_run": False,
            "framework_name": "pytest",
        },
    )
    assert exec_res.status_code == 200, f"Docker execution failed: {exec_res.text}"
    exec_data = exec_res.json()
    exec_id = exec_data["execution_run_id"]

    # Verify real Docker container execution evidence
    assert exec_data["is_docker_isolated"] is True
    assert exec_data["docker_diagnostics"]["is_available"] is True
    assert exec_data["total_executed"] > 0
    assert len(exec_data["results"]) > 0

    # Inspect container results
    for r in exec_data["results"]:
        assert r["duration_ms"] > 0
        assert r["exit_code"] in (0, 1)
        # Verify container execution output captured
        assert len(r["stdout"]) > 0 or len(r["stderr"]) > 0

    # Verify workflow state updated
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.execution_run_id == exec_id
    assert wf.stages["execution"].status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS)

    # 6. Stage 4: Evidence-Based Failure Arbitration
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={"workflow_id": wf_id, "execution_run_id": exec_id},
    )
    assert arb_res.status_code == 200
    arb_data = arb_res.json()
    arb_id = arb_data["arbitration_run_id"]

    assert arb_id.startswith("arb-")
    assert "category_counts" in arb_data
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.stages["arbitration"].status == StageStatus.COMPLETED
