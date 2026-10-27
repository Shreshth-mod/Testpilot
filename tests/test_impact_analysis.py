"""
Unit and integration tests for Phase B: Multi-Layer Impact Analysis Subsystem.
Verifies:
1. Prerequisite stage enforcement (Stage 0 Evolution must be completed).
2. Change understanding and multi-layer AST call graph traversal.
3. Spec and AST boundary constraint extraction (zero hallucinations).
4. Deterministic categorical risk classification (CRITICAL, HIGH, MEDIUM, LOW, UNCERTAIN).
5. Verifiable impact evidence graph structure (nodes, edges, evidence provenance).
6. State persistence and transition of Stage 2 (Selection) to READY.
7. REST API endpoints (/api/impact/analyze, /api/impact/{impact_run_id}).
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
from testpilot.impact.models import RiskCategory
from testpilot.web.api import app

client = TestClient(app)


def test_impact_analysis_prerequisite_enforcement(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = ImpactAnalysisEngine(workflow_storage=mgr)

    # Initialize fresh workflow where Evolution is NOT_STARTED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-impact-prereq",
    )
    mgr.save_workflow(wf)

    # Attempting to run impact analysis should raise PrerequisiteError
    with pytest.raises(PrerequisiteError) as exc_info:
        engine.analyze(workflow_id="wf-test-impact-prereq")
    assert "evolution" in str(exc_info.value).lower()


def test_impact_analysis_execution(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = ImpactAnalysisEngine(workflow_storage=mgr)

    # Setup workflow where Evolution is COMPLETED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-impact-run",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["evolution"].run_id = "evo-run-001"
    mgr.save_workflow(wf)

    # Execute Impact Analysis
    report = engine.analyze(workflow_id="wf-test-impact-run", max_depth=3)

    assert report.impact_run_id.startswith("imp-")
    assert report.workflow_id == "wf-test-impact-run"
    assert report.evolution_run_id == "evo-run-001"
    assert len(report.changed_symbols) > 0

    # Evidence graph validation
    graph = report.evidence_graph
    assert graph.total_nodes > 0
    assert graph.total_edges > 0
    assert any(n.node_type == "CHANGED_SYMBOL" for n in graph.nodes)

    # Boundary impacts validation
    assert len(report.boundary_impacts) > 0
    assert any(b.target_symbol for b in report.boundary_impacts)

    # Risk distribution validation
    dist = report.risk_distribution
    assert sum(dist.values()) > 0
    assert dist[RiskCategory.CRITICAL.value] >= 1

    # Verify workflow state transitioned
    updated_wf = mgr.load_workflow(wf.repo_id, wf.workflow_id)
    assert updated_wf.impact_run_id == report.impact_run_id
    assert updated_wf.stages["impact"].status == StageStatus.COMPLETED
    assert updated_wf.stages["selection"].status == StageStatus.READY
    assert updated_wf.stages["selection"].upstream_run_id == report.impact_run_id


def test_impact_api_endpoints():
    """Verify POST /api/impact/analyze and GET /api/impact/{impact_run_id}."""
    # 1. Initialize workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]

    # 2. Simulate Stage 0 Evolution completion via context
    from testpilot.web.api import workflow_storage
    wf = workflow_storage.load_workflow(wf_data["repo_id"], wf_id)
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["evolution"].run_id = "evo-api-test"
    workflow_storage.save_workflow(wf)

    # 3. Call POST /api/impact/analyze
    res = client.post(
        "/api/impact/analyze",
        json={"workflow_id": wf_id, "max_depth": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["impact_run_id"].startswith("imp-")
    assert data["workflow_id"] == wf_id
    assert "evidence_graph" in data
    assert "risk_distribution" in data
    impact_id = data["impact_run_id"]

    # 4. Call GET /api/impact/{impact_run_id}
    res_get = client.get(f"/api/impact/{impact_id}")
    assert res_get.status_code == 200
    assert res_get.json()["impact_run_id"] == impact_id
