"""
Unit and integration tests for Phase D: Isolated Test Execution with Docker (Stage 3).
Verifies:
1. Prerequisite gating (Stage 2 Selection must be COMPLETED).
2. Unsupported framework rejection (explicit error without guessing commands).
3. Docker availability diagnostics and setup instructions.
4. Unsafe host execution rejection when Docker daemon is stopped.
5. Sandboxed dry-run execution and test telemetry collection.
6. Workflow state transition (Stage 3 COMPLETED, Stage 4 READY).
7. REST API endpoints (/api/execution/docker/status, /api/execution/run, /api/execution/{id}).
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
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import TestOutcome
from testpilot.execution.runner import (
    DockerUnavailableError,
    PytestDockerRunner,
    UnsupportedFrameworkError,
)
from testpilot.impact.engine import ImpactAnalysisEngine
from testpilot.selection.engine import TestSelectionEngine
from testpilot.web.api import app

client = TestClient(app)


def test_execution_prerequisite_enforcement(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = IsolatedExecutionEngine(workflow_storage=mgr)

    # Workflow where Selection is NOT_STARTED
    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-exec-prereq",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    with pytest.raises(PrerequisiteError) as exc_info:
        engine.execute(workflow_id="wf-test-exec-prereq")
    assert "selection" in str(exc_info.value).lower()


def test_unsupported_framework_rejection(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = IsolatedExecutionEngine(workflow_storage=mgr)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-framework",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    wf.selection_run_id = "sel-dummy"
    mgr.save_workflow(wf)

    with pytest.raises(UnsupportedFrameworkError) as exc_info:
        engine.execute(workflow_id="wf-test-framework", framework_name="maven_surefire")
    assert "unsupported" in str(exc_info.value).lower()


def test_docker_diagnostics_and_safety_rejection(tmp_path):
    runner = PytestDockerRunner()
    diag = runner.check_docker_availability()
    assert isinstance(diag.is_available, bool)
    assert len(diag.setup_guidance) >= 3

    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = IsolatedExecutionEngine(workflow_storage=mgr, runner=runner)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-safety",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    wf.selection_run_id = "sel-dummy"
    mgr.save_workflow(wf)

    # If Docker daemon is stopped, execution must fail safely with DockerUnavailableError
    if not diag.is_available:
        with pytest.raises(DockerUnavailableError) as exc_info:
            engine.execute(workflow_id="wf-test-safety", allow_simulated_dry_run=False)
        assert "docker daemon unavailable" in str(exc_info.value).lower()
        assert "unsafe local host execution is rejected" in str(exc_info.value).lower()


def test_sandboxed_dry_run_execution(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    engine = IsolatedExecutionEngine(workflow_storage=mgr)

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=".",
        repo_name="Testbed",
        base_commit_hash="1" * 40,
        target_commit_hash="2" * 40,
        workflow_id="wf-test-dry-run",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    wf.selection_run_id = "sel-dummy"
    mgr.save_workflow(wf)

    # Run with allow_simulated_dry_run=True
    report = engine.execute(workflow_id="wf-test-dry-run", allow_simulated_dry_run=True)
    assert report.execution_run_id.startswith("exec-")
    assert report.total_executed >= 1
    assert len(report.results) >= 1

    # Check workflow state transitioned to Stage 4 READY
    updated_wf = mgr.load_workflow(wf.repo_id, wf.workflow_id)
    assert updated_wf.execution_run_id == report.execution_run_id
    assert updated_wf.stages["execution"].status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS)
    assert updated_wf.stages["arbitration"].status == StageStatus.READY
    assert updated_wf.stages["arbitration"].upstream_run_id == report.execution_run_id


def test_execution_api_endpoints():
    """Verify REST API endpoints: /api/execution/docker/status, /api/execution/run, /api/execution/{id}."""
    # 1. Test docker status endpoint
    status_res = client.get("/api/execution/docker/status")
    assert status_res.status_code == 200
    diag = status_res.json()
    assert "is_available" in diag
    assert "setup_guidance" in diag

    # 2. Initialize workflow and complete prior stages
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD", "allow_dirty": True},
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]

    from testpilot.web.api import workflow_storage
    wf = workflow_storage.load_workflow(wf_data["repo_id"], wf_id)
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["selection"].status = StageStatus.COMPLETED
    wf.selection_run_id = "sel-api-test"
    workflow_storage.save_workflow(wf)

    # 3. Post to /api/execution/run with allow_simulated_dry_run=True
    run_res = client.post(
        "/api/execution/run",
        json={"workflow_id": wf_id, "allow_simulated_dry_run": True},
    )
    assert run_res.status_code == 200
    exec_data = run_res.json()
    assert exec_data["execution_run_id"].startswith("exec-")
    assert "results" in exec_data
    exec_id = exec_data["execution_run_id"]

    # 4. Get by ID
    get_res = client.get(f"/api/execution/{exec_id}")
    assert get_res.status_code == 200
    assert get_res.json()["execution_run_id"] == exec_id
