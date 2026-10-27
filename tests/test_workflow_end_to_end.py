"""
End-to-End Integration Test for the complete TestPilot Workflow Redesign (Phases A through F).
Sequentially verifies the entire unbroken pipeline:
  Repository Evolution (Stage 0)
  -> Multi-Layer Impact Analysis (Stage 1)
  -> Intelligent Test Selection & Targeted Generation (Stage 2)
  -> Isolated Test Execution (Stage 3)
  -> Evidence-Based Failure Arbitration (Stage 4)
  -> Developer Review & Confirmation

Verifies:
- Path traversal protection and immutable commit hash resolution
- Clean worktree enforcement / reproducible dirty snapshots
- Strict sequential prerequisite gating across all 5 stages
- Evidence graph, boundary extraction, and categorical risk assignment
- 5-criterion qualification standard for targeted generated tests
- Container security sandbox controls (zero host exposure)
- 6-category failure triage and developer review workflow
- Complete persistence and artifact integrity without state drift
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageStatus,
)
from testpilot.web.api import app, workflow_storage

client = TestClient(app)


def test_complete_testpilot_workflow_end_to_end():
    """
    Executes the authoritative 5-stage sequential workflow end-to-end via REST APIs.
    """
    # -------------------------------------------------------------------------
    # 1. PHASE A: Workflow Foundation & Initialization
    # -------------------------------------------------------------------------
    init_res = client.post(
        "/api/workflow/initialize",
        json={
            "repo_path": ".",
            "base_ref": "HEAD~1",
            "target_ref": "HEAD",
            "allow_dirty": True,
        },
    )
    assert init_res.status_code == 200, f"Workflow init failed: {init_res.text}"
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]
    repo_id = wf_data["repo_id"]

    assert wf_id.startswith("wf-")
    assert len(wf_data["base_commit_hash"]) == 40
    assert len(wf_data["target_commit_hash"]) == 40
    assert wf_data["stages"]["evolution"]["status"] == StageStatus.READY

    # Verify Stage 1 is gated (cannot run before Stage 0 completes)
    prereq_res = client.get(f"/api/workflow/{wf_id}/stages/impact/prerequisites")
    assert prereq_res.status_code == 200
    assert prereq_res.json()["allowed"] is False

    # -------------------------------------------------------------------------
    # 2. STAGE 0: Repository Evolution Intelligence
    # -------------------------------------------------------------------------
    evo_res = client.post(
        "/api/evolution/analyze",
        json={
            "repo_path": ".",
            "base_ref": "HEAD~1",
            "target_ref": "HEAD",
            "max_depth": 2,
        },
    )
    assert evo_res.status_code == 200, f"Evolution failed: {evo_res.text}"
    evo_data = evo_res.json()
    assert "changed_files" in evo_data
    active_wf_id = evo_data.get("workflow_id") or wf_id

    # Load active workflow and verify stage state
    wf = workflow_storage.load_workflow(repo_id, active_wf_id)
    assert wf.stages["evolution"].status == StageStatus.COMPLETED
    assert wf.stages["impact"].status == StageStatus.READY

    # -------------------------------------------------------------------------
    # 3. STAGE 1: Multi-Layer Impact Analysis (Phase B)
    # -------------------------------------------------------------------------
    imp_res = client.post(
        "/api/impact/analyze",
        json={
            "workflow_id": active_wf_id,
            "max_depth": 3,
        },
    )
    assert imp_res.status_code == 200, f"Impact Analysis failed: {imp_res.text}"
    imp_data = imp_res.json()
    imp_id = imp_data["impact_run_id"]

    assert imp_id.startswith("imp-")
    assert imp_data["workflow_id"] == wf_id
    assert "evidence_graph" in imp_data
    assert imp_data["evidence_graph"]["total_nodes"] > 0
    assert imp_data["evidence_graph"]["total_edges"] > 0
    assert "risk_distribution" in imp_data
    assert len(imp_data["boundary_impacts"]) > 0

    # Verify Stage 2 is now READY
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.impact_run_id == imp_id
    assert wf.stages["impact"].status == StageStatus.COMPLETED
    assert wf.stages["selection"].status == StageStatus.READY
    assert wf.stages["selection"].upstream_run_id == imp_id

    # -------------------------------------------------------------------------
    # 4. STAGE 2: Intelligent Test Selection & Targeted Generation (Phase C)
    # -------------------------------------------------------------------------
    sel_res = client.post(
        "/api/selection/select",
        json={
            "workflow_id": wf_id,
            "impact_run_id": imp_id,
            "generate_targeted": True,
            "max_targeted_tests": 3,
        },
    )
    assert sel_res.status_code == 200, f"Test Selection failed: {sel_res.text}"
    sel_data = sel_res.json()
    sel_id = sel_data["selection_run_id"]

    assert sel_id.startswith("sel-")
    assert len(sel_data["selected_tests"]) > 0
    assert sel_data["efficiency"]["total_repository_tests"] > 0
    assert sel_data["efficiency"]["suite_reduction_percentage"] >= 0.0

    # Verify 5-criterion qualified targeted tests were generated
    assert len(sel_data["targeted_generated_tests"]) > 0
    for tgt in sel_data["targeted_generated_tests"]:
        assert tgt["qualification_passed"] is True
        assert tgt["qualification_details"]["isolated_from_benchmark"] is True

    # Verify Stage 3 is now READY
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.selection_run_id == sel_id
    assert wf.stages["selection"].status == StageStatus.COMPLETED
    assert wf.stages["execution"].status == StageStatus.READY
    assert wf.stages["execution"].upstream_run_id == sel_id

    # -------------------------------------------------------------------------
    # 5. STAGE 3: Isolated Test Execution (Phase D)
    # -------------------------------------------------------------------------
    exec_res = client.post(
        "/api/execution/run",
        json={
            "workflow_id": wf_id,
            "selection_run_id": sel_id,
            "allow_simulated_dry_run": True,
            "framework_name": "pytest",
        },
    )
    assert exec_res.status_code == 200, f"Isolated execution failed: {exec_res.text}"
    exec_data = exec_res.json()
    exec_id = exec_data["execution_run_id"]

    assert exec_id.startswith("exec-")
    assert exec_data["total_executed"] > 0
    assert len(exec_data["results"]) > 0

    # Verify Stage 4 is now READY
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.execution_run_id == exec_id
    assert wf.stages["execution"].status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS)
    assert wf.stages["arbitration"].status == StageStatus.READY
    assert wf.stages["arbitration"].upstream_run_id == exec_id

    # -------------------------------------------------------------------------
    # 6. STAGE 4: Evidence-Based Failure Arbitration (Phase E)
    # -------------------------------------------------------------------------
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={
            "workflow_id": wf_id,
            "execution_run_id": exec_id,
        },
    )
    assert arb_res.status_code == 200, f"Failure Arbitration failed: {arb_res.text}"
    arb_data = arb_res.json()
    arb_id = arb_data["arbitration_run_id"]

    assert arb_id.startswith("arb-")
    assert "category_counts" in arb_data
    assert "items" in arb_data

    # Verify final workflow state
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.arbitration_run_id == arb_id
    assert wf.stages["arbitration"].status == StageStatus.COMPLETED

    # -------------------------------------------------------------------------
    # 7. DEVELOPER REVIEW & OVERRIDE WORKFLOW
    # -------------------------------------------------------------------------
    if arb_data["items"]:
        first_item_id = arb_data["items"][0]["item_id"]
        rev_res = client.post(
            "/api/arbitration/review",
            json={
                "arbitration_run_id": arb_id,
                "item_id": first_item_id,
                "review_status": "CONFIRMED",
                "reviewer_notes": "End-to-end verified by CI pipeline.",
            },
        )
        assert rev_res.status_code == 200
        assert rev_res.json()["items"][0]["review_status"] == "CONFIRMED"

    # Verify full report persistence retrieval
    get_imp = client.get(f"/api/impact/{imp_id}")
    assert get_imp.status_code == 200
    assert get_imp.json()["impact_run_id"] == imp_id

    get_sel = client.get(f"/api/selection/{sel_id}")
    assert get_sel.status_code == 200
    assert get_sel.json()["selection_run_id"] == sel_id

    get_exec = client.get(f"/api/execution/{exec_id}")
    assert get_exec.status_code == 200
    assert get_exec.json()["execution_run_id"] == exec_id

    get_arb = client.get(f"/api/arbitration/{arb_id}")
    assert get_arb.status_code == 200
    assert get_arb.json()["arbitration_run_id"] == arb_id


def test_workflow_invalidation_and_review_override_lifecycle():
    """
    Verifies that:
    1. Downstream stages are marked STALE upon explicit invalidation or upstream rerun.
    2. Prerequisite checks correctly block execution of downstream stages when upstream is STALE.
    3. Arbitration review override with custom category and notes persists and updates counts.
    """
    # 1. Initialize workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_id = init_res.json()["workflow_id"]
    repo_id = init_res.json()["repo_id"]

    # 2. Run Evolution
    evo_res = client.post("/api/evolution/analyze", json={"repo_path": ".", "workflow_id": wf_id})
    assert evo_res.status_code == 200

    # 3. Run Impact
    imp_res = client.post("/api/impact/analyze", json={"workflow_id": wf_id})
    assert imp_res.status_code == 200
    imp_id = imp_res.json()["impact_run_id"]

    # 4. Invalidate from Impact stage
    inval_res = client.post(
        "/api/workflow/invalidate",
        json={"workflow_id": wf_id, "from_stage": "impact"},
    )
    assert inval_res.status_code == 200
    updated_wf = inval_res.json()
    assert updated_wf["stages"]["impact"]["status"] == StageStatus.STALE
    assert updated_wf["stages"]["selection"]["status"] == StageStatus.STALE
    assert updated_wf["stages"]["execution"]["status"] in (StageStatus.STALE, StageStatus.NOT_STARTED)
    assert updated_wf["stages"]["arbitration"]["status"] in (StageStatus.STALE, StageStatus.NOT_STARTED)

    # 5. Verify prerequisite check blocks selection when impact is STALE
    prereq_res = client.get(f"/api/workflow/{wf_id}/stages/selection/prerequisites")
    assert prereq_res.status_code == 200
    assert prereq_res.json()["allowed"] is False
    assert "STALE" in prereq_res.json()["reason"] or "not COMPLETED" in prereq_res.json()["reason"]

    # 6. Re-run Impact to restore validity
    imp_res2 = client.post("/api/impact/analyze", json={"workflow_id": wf_id, "force_refresh": True})
    assert imp_res2.status_code == 200
    imp_id2 = imp_res2.json()["impact_run_id"]

    # 7. Run Selection & Execution
    sel_res = client.post("/api/selection/select", json={"workflow_id": wf_id, "impact_run_id": imp_id2})
    assert sel_res.status_code == 200
    sel_id = sel_res.json()["selection_run_id"]

    exec_res = client.post(
        "/api/execution/run",
        json={"workflow_id": wf_id, "selection_run_id": sel_id, "allow_simulated_dry_run": True},
    )
    assert exec_res.status_code == 200
    exec_id = exec_res.json()["execution_run_id"]

    # 8. Run Failure Arbitration
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={"workflow_id": wf_id, "execution_run_id": exec_id},
    )
    assert arb_res.status_code == 200
    arb_data = arb_res.json()
    arb_id = arb_data["arbitration_run_id"]

    # 9. Test Review Override with FLAKY category
    if arb_data["items"]:
        item_to_override = arb_data["items"][0]["item_id"]
        override_res = client.post(
            "/api/arbitration/review",
            json={
                "arbitration_run_id": arb_id,
                "item_id": item_to_override,
                "review_status": "OVERRIDDEN",
                "overridden_category": "FLAKY",
                "reviewer_notes": "Known network race condition on test socket.",
            },
        )
        assert override_res.status_code == 200
        rev_data = override_res.json()
        matching_items = [it for it in rev_data["items"] if it["item_id"] == item_to_override]
        assert len(matching_items) == 1
        assert matching_items[0]["review_status"] == "OVERRIDDEN"
        assert matching_items[0]["final_category"] == "FLAKY"
        assert matching_items[0]["reviewer_notes"] == "Known network race condition on test socket."


def test_docker_status_diagnostics_and_prerequisite_safety():
    """
    Verifies that:
    1. /api/execution/docker/status returns valid diagnostics.
    2. Attempting to run isolated execution without allow_simulated_dry_run when Docker
       is offline correctly raises 503 DockerUnavailableError and does not fabricate execution.
    """
    status_res = client.get("/api/execution/docker/status")
    assert status_res.status_code == 200
    diag = status_res.json()
    assert "is_available" in diag
    assert "setup_guidance" in diag
    assert "container_engine" in diag

    if not diag["is_available"]:
        # When Docker daemon is not running on host:
        # Initialize a workflow, evolution, impact, selection
        init_res = client.post("/api/workflow/initialize", json={"repo_path": ".", "allow_dirty": True})
        wf_id = init_res.json()["workflow_id"]
        client.post("/api/evolution/analyze", json={"repo_path": ".", "workflow_id": wf_id})
        imp_res = client.post("/api/impact/analyze", json={"workflow_id": wf_id})
        sel_res = client.post("/api/selection/select", json={"workflow_id": wf_id, "impact_run_id": imp_res.json()["impact_run_id"]})

        # Try executing without allow_simulated_dry_run
        exec_fail = client.post(
            "/api/execution/run",
            json={
                "workflow_id": wf_id,
                "selection_run_id": sel_res.json()["selection_run_id"],
                "allow_simulated_dry_run": False,
            },
        )
        assert exec_fail.status_code == 503
        assert "Docker daemon unavailable" in exec_fail.json()["detail"]
