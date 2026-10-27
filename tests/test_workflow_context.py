"""
Unit and integration tests for Phase A: Workflow Context & Lifecycle Subsystem.
Verifies:
1. Path traversal security sanitization for repository identifiers.
2. Atomic multi-run workflow context persistence and recovery.
3. Immutable commit hash resolution (symbolic ref -> 40-char SHA).
4. Working tree clean verification and reproducible dirty snapshotting.
5. Strict sequential stage prerequisites and critical warning gating.
6. Downstream invalidation into STALE state on drift.
7. REST API workflow endpoints (/api/workflow/*).
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageMetadata,
    StageStatus,
    SecurityViolationError,
    PrerequisiteError,
    STAGE_SEQUENCE,
)
from testpilot.web.api import app

client = TestClient(app)


def test_sanitize_repo_id_prevents_path_traversal(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")

    # Path traversal attempts
    traversal_input = "../../etc/passwd"
    safe_id = mgr.sanitize_repo_id(traversal_input, "/some/path")
    assert ".." not in safe_id
    assert "/" not in safe_id
    assert "\\" not in safe_id

    # Windows traversal attempts
    win_traversal = "..\\..\\Windows\\System32"
    safe_win_id = mgr.sanitize_repo_id(win_traversal, "C:\\some\\path")
    assert ".." not in safe_win_id
    assert "\\" not in safe_win_id

    # Validated storage path resolution
    storage_path = mgr.get_workflow_path(safe_id, "wf-test-123")
    assert storage_path.is_relative_to(mgr.base_dir)


def test_atomic_workflow_save_and_load(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    repo_id = "test_repo"

    wf = WorkflowContext.create_initial(
        repo_id=repo_id,
        repo_path=str(tmp_path),
        repo_name="Test Repo",
        base_commit_hash="a" * 40,
        target_commit_hash="b" * 40,
        workflow_id="wf-unit-001",
    )

    mgr.save_workflow(wf)

    # Verify file was written
    target_file = mgr.get_workflow_path(repo_id, "wf-unit-001")
    assert target_file.exists()
    assert not target_file.with_suffix(".json.tmp").exists()

    # Load back
    loaded = mgr.load_workflow(repo_id, "wf-unit-001")
    assert loaded is not None
    assert loaded.workflow_id == "wf-unit-001"
    assert loaded.base_commit_hash == "a" * 40
    assert loaded.target_commit_hash == "b" * 40
    assert loaded.stages["evolution"].status == StageStatus.READY


def test_resolve_immutable_commit_sha():
    mgr = WorkflowStorageManager()
    # Resolve HEAD on current repository
    sha = mgr.resolve_immutable_commit(".", "HEAD")
    assert len(sha) == 40
    assert all(c in "0123456789abcdefABCDEF" for c in sha)

    # Invalid ref raises ValueError
    with pytest.raises(ValueError):
        mgr.resolve_immutable_commit(".", "nonexistent_ref_123456789_xyz")


def test_working_tree_clean_and_snapshot(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    is_clean, status_txt = mgr.check_working_tree_clean(".")
    assert isinstance(is_clean, bool)

    # Creating snapshot generates valid manifest and sha256
    snap_id, snap_hash = mgr.create_dirty_snapshot(".", "wf-test-snap")
    assert snap_id.startswith("snap-")
    assert len(snap_hash) == 64  # SHA-256


def test_stage_sequence_and_prerequisites(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=str(tmp_path),
        repo_name="Test Repo",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-prereq-001",
    )

    # Stage 0: Evolution can always run
    allowed, reason = mgr.validate_stage_transition(wf, "evolution")
    assert allowed is True

    # Stage 1: Impact cannot run before Evolution completes
    allowed, reason = mgr.validate_stage_transition(wf, "impact")
    assert allowed is False
    assert "evolution" in reason

    # Mark Evolution COMPLETED
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["evolution"].run_id = "evo-run-1"
    allowed, reason = mgr.validate_stage_transition(wf, "impact")
    assert allowed is True

    # If Evolution completed with critical warnings, progression is blocked
    wf.stages["evolution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.stages["evolution"].critical_warnings = ["AST parse failed on critical target file"]
    allowed, reason = mgr.validate_stage_transition(wf, "impact")
    assert allowed is False
    assert "critical warnings" in reason.lower()

    # If warnings are non-critical, progression is allowed
    wf.stages["evolution"].critical_warnings = []
    wf.stages["evolution"].warnings = ["Minor typing hint missing in optional file"]
    allowed, reason = mgr.validate_stage_transition(wf, "impact")
    assert allowed is True


def test_downstream_invalidation_to_stale(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=str(tmp_path),
        repo_name="Test Repo",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-inval-001",
    )

    # Complete all stages
    for stage_name in STAGE_SEQUENCE:
        wf.stages[stage_name].status = StageStatus.COMPLETED
        wf.stages[stage_name].run_id = f"{stage_name}-1"

    # Invalidate starting from impact (Stage 1)
    updated = mgr.invalidate_downstream(wf, "impact")

    # Evolution remains COMPLETED
    assert updated.stages["evolution"].status == StageStatus.COMPLETED

    # Impact, Selection, Execution, Arbitration must be STALE
    assert updated.stages["impact"].status == StageStatus.STALE
    assert updated.stages["selection"].status == StageStatus.STALE
    assert updated.stages["execution"].status == StageStatus.STALE
    assert updated.stages["arbitration"].status == StageStatus.STALE


def test_workflow_rest_api_active():
    """Verify GET /api/workflow/active returns a valid WorkflowContext model."""
    res = client.get("/api/workflow/active")
    assert res.status_code == 200
    data = res.json()
    assert "workflow_id" in data
    assert "repo_id" in data
    assert "base_commit_hash" in data
    assert "target_commit_hash" in data
    assert "stages" in data
    assert len(data["base_commit_hash"]) == 40
    assert len(data["target_commit_hash"]) == 40


def test_workflow_rest_api_initialize():
    """Verify POST /api/workflow/initialize creates an authoritative context."""
    res = client.post(
        "/api/workflow/initialize",
        json={
            "repo_path": ".",
            "base_ref": "HEAD~1",
            "target_ref": "HEAD",
            "allow_dirty": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["workflow_id"].startswith("wf-")
    assert len(data["base_commit_hash"]) == 40
    assert len(data["target_commit_hash"]) == 40
    wf_id = data["workflow_id"]

    # Verify GET by ID
    res2 = client.get(f"/api/workflow/{wf_id}")
    assert res2.status_code == 200
    assert res2.json()["workflow_id"] == wf_id

    # Verify prerequisite check endpoint
    res3 = client.get(f"/api/workflow/{wf_id}/stages/impact/prerequisites")
    assert res3.status_code == 200
    prereq = res3.json()
    assert prereq["workflow_id"] == wf_id
    assert prereq["stage"] == "impact"
    assert prereq["allowed"] is False  # evolution not yet completed

    # Verify invalidation endpoint
    res4 = client.post(
        "/api/workflow/invalidate",
        json={"workflow_id": wf_id, "from_stage": "impact"},
    )
    assert res4.status_code == 200
