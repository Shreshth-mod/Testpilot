"""
Comprehensive End-to-End Acceptance and Workflow Verification for TestPilot:
- Stage 2: Affected file analysis, test selection, targeted test generation & qualification
- Stage 3: Real isolated Docker execution (enforces allow_simulated_dry_run=False, verifies no host fallback)
- Stage 4: Evidence-based failure arbitration with prerequisite gates and state machine validation
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from testpilot.core.workflow_context import StageStatus, WorkflowContext
from testpilot.execution.models import DockerDiagnostics
from testpilot.execution.runner import PytestDockerRunner
from testpilot.web.api import app, workflow_storage

client = TestClient(app)


def _setup_controlled_repo(repo_dir: Path) -> tuple[str, str]:
    """
    Constructs a genuine multi-file Python repository with git version control.
    Base commit: CartItem schema and OrderService caller.
    Target commit: Multi-file change updating CartItem discount boundary and OrderService subtotal logic.
    """
    # 1. Initialize Git repository
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestPilot Engineer"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "engineer@testpilot.ai"], cwd=repo_dir, check=True, capture_output=True)

    # 2. Write pyproject.toml
    (repo_dir / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "controlled-order-app"
version = "0.1.0"
dependencies = ["pydantic>=2.0.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
""",
        encoding="utf-8",
    )

    # 3. Create app package
    app_dir = repo_dir / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("", encoding="utf-8")

    # Base CartItem model with strict positive boundary gt=0.0
    models_file = app_dir / "models.py"
    models_file.write_text(
        """from pydantic import BaseModel, Field

class CartItem(BaseModel):
    item_id: str = "item-1"
    name: str = "Test Item"
    unit_price: float = Field(..., gt=0.0, description="Price per unit")
    quantity: int = Field(default=1, gt=0, le=1000)
""",
        encoding="utf-8",
    )

    # Base OrderService caller
    services_dir = app_dir / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    (services_dir / "__init__.py").write_text("", encoding="utf-8")

    order_svc_file = services_dir / "order_service.py"
    order_svc_file.write_text(
        """from app.models import CartItem

class OrderService:
    @staticmethod
    def calculate_subtotal(items: list[CartItem]) -> float:
        subtotal = sum(item.unit_price * item.quantity for item in items)
        return round(subtotal, 2)
""",
        encoding="utf-8",
    )

    # Existing regression test
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    test_file = tests_dir / "test_order_service.py"
    test_file.write_text(
        """from app.models import CartItem
from app.services.order_service import OrderService

def test_calculate_subtotal_nominal():
    items = [CartItem(item_id="it-1", unit_price=10.0, quantity=2)]
    total = OrderService.calculate_subtotal(items)
    assert total == 20.0
""",
        encoding="utf-8",
    )

    # Commit base revision
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "feat: initial order service implementation"], cwd=repo_dir, check=True, capture_output=True)
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    # 4. Create Multi-File Target Change
    # Change 1: models.py adds discount_rate boundary (ge=0.0, le=1.0)
    models_file.write_text(
        """from pydantic import BaseModel, Field

class CartItem(BaseModel):
    item_id: str = "item-1"
    name: str = "Test Item"
    unit_price: float = Field(..., gt=0.0, description="Price per unit")
    quantity: int = Field(default=1, gt=0, le=1000)
    discount_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Item discount rate")
""",
        encoding="utf-8",
    )

    # Change 2: services/order_service.py updates calculate_subtotal to account for discount_rate
    order_svc_file.write_text(
        """from app.models import CartItem

class OrderService:
    @staticmethod
    def calculate_subtotal(items: list[CartItem]) -> float:
        subtotal = sum(
            item.unit_price * (1.0 - getattr(item, "discount_rate", 0.0)) * item.quantity
            for item in items
        )
        return round(subtotal, 2)
""",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "feat: support item discount rate across models and service caller"], cwd=repo_dir, check=True, capture_output=True)
    target_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    return base_sha, target_sha


def test_docker_unavailable_rejection_scenario(tmp_path: Path):
    """
    Verifies that when Docker is unavailable:
    1. Unsafe host execution is rejected (never silently falls back).
    2. Stage 3 returns descriptive diagnostics and does not transition to COMPLETED.
    3. Stage 4 failure arbitration remains strictly blocked by prerequisite enforcement.
    """
    repo_dir = tmp_path / "controlled_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    base_sha, target_sha = _setup_controlled_repo(repo_dir)

    # Initialize workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": str(repo_dir), "base_ref": base_sha, "target_ref": target_sha},
    )
    assert init_res.status_code == 200
    wf_id = init_res.json()["workflow_id"]

    # Stage 0 & 1
    evo_res = client.post("/api/evolution/analyze", json={"repo_path": str(repo_dir), "workflow_id": wf_id})
    assert evo_res.status_code == 200
    imp_res = client.post("/api/impact/analyze", json={"workflow_id": wf_id, "max_depth": 2})
    assert imp_res.status_code == 200

    # Stage 2
    sel_res = client.post(
        "/api/selection/select",
        json={"workflow_id": wf_id, "generate_targeted": True, "max_targeted_tests": 1},
    )
    assert sel_res.status_code == 200
    sel_id = sel_res.json()["selection_run_id"]

    # Stage 3 with Simulated Docker Daemon Unavailable
    with patch.object(
        PytestDockerRunner,
        "check_docker_availability",
        return_value=DockerDiagnostics(
            is_available=False,
            error_message="Docker daemon is not running or not responsive.",
        ),
    ):
        exec_res = client.post(
            "/api/execution/run",
            json={
                "workflow_id": wf_id,
                "selection_run_id": sel_id,
                "allow_simulated_dry_run": False,
            },
        )
        # Execution must fail and reject unsafe local host execution
        assert exec_res.status_code in (400, 500, 503)
        assert "Docker daemon unavailable" in exec_res.text or "Unsafe local host execution is rejected" in exec_res.text

    # Verify Stage 4 is BLOCKED because Stage 3 is not completed
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={"workflow_id": wf_id},
    )
    assert arb_res.status_code in (400, 412)
    assert "Stage 'execution' must be completed" in arb_res.text or "Prerequisite failure" in arb_res.text


def test_real_docker_controlled_multi_file_lifecycle(tmp_path: Path):
    """
    Verifies the complete real execution lifecycle through Stages 2, 3, and 4
    using a controlled Python repository with a genuine multi-file change (CartItem & OrderService).
    """
    repo_dir = tmp_path / "controlled_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    base_sha, target_sha = _setup_controlled_repo(repo_dir)

    # 0. Confirm Docker Daemon and Image Availability
    status_res = client.get("/api/execution/docker/status")
    assert status_res.status_code == 200
    diag = status_res.json()
    assert diag["is_available"] is True, f"Docker daemon must be online: {diag.get('error_message')}"

    # 1. Initialize Workflow
    init_res = client.post(
        "/api/workflow/initialize",
        json={"repo_path": str(repo_dir), "base_ref": base_sha, "target_ref": target_sha},
    )
    assert init_res.status_code == 200
    wf_data = init_res.json()
    wf_id = wf_data["workflow_id"]
    repo_id = wf_data["repo_id"]

    # 2. Stage 0: Repository Evolution (Diff & AST Extraction)
    evo_res = client.post("/api/evolution/analyze", json={"repo_path": str(repo_dir), "workflow_id": wf_id})
    assert evo_res.status_code == 200
    evo_data = evo_res.json()
    assert len(evo_data["changed_files"]) == 2
    assert any("models.py" in f for f in evo_data["changed_files"])
    assert any("order_service.py" in f for f in evo_data["changed_files"])

    # 3. Stage 1: Impact Analysis (Reachability & Risk)
    imp_res = client.post("/api/impact/analyze", json={"workflow_id": wf_id, "max_depth": 2})
    assert imp_res.status_code == 200
    imp_data = imp_res.json()
    imp_id = imp_data["impact_run_id"]
    assert len(imp_data["impacted_test_candidates"]) > 0

    # 4. Stage 2: Intelligent Test Selection & Targeted Generation
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

    # Verify Stage 2 Affected Files Analysis
    assert len(sel_data["affected_files"]) >= 2
    types_found = {af["change_type"] for af in sel_data["affected_files"]}
    assert "validation_boundary" in types_found
    assert "business_logic" in types_found

    # Verify Stage 2 Selected Existing Regression Tests
    assert sel_data["existing_tests_selected_count"] >= 1
    selected_tests = sel_data["selected_tests"]
    assert any(st["test_name"] == "test_calculate_subtotal_nominal" for st in selected_tests)

    # Verify Stage 2 Targeted Generated Tests & Qualification (5-criterion)
    assert sel_data["new_tests_generated_count"] >= 1
    assert sel_data["tests_qualified_count"] >= 1
    gen_test = sel_data["targeted_generated_tests"][0]
    assert gen_test["qualification_passed"] is True
    assert gen_test["qualification_details"]["ast_syntax_valid"] is True
    assert gen_test["qualification_details"]["collectable"] is True
    assert gen_test["qualification_details"]["has_non_trivial_assertions"] is True
    assert gen_test["qualification_details"]["isolated_from_benchmark"] is True

    # Verify Targeted Tests are Persisted to Job-Specific File
    target_file = (repo_dir / gen_test["file_path"]) if not Path(gen_test["file_path"]).is_absolute() else Path(gen_test["file_path"])
    assert target_file.exists(), f"Targeted test file must be written to disk: {target_file}"
    content = target_file.read_text(encoding="utf-8")
    assert "def test_" in content

    # Verify Initial Status is NOT_RUN
    assert gen_test["execution_status"] == "NOT_RUN"
    assert sel_data["tests_executed_count"] == 0

    # 5. Stage 3: Real Isolated Docker Execution (Enforces allow_simulated_dry_run=False)
    exec_res = client.post(
        "/api/execution/run",
        json={
            "workflow_id": wf_id,
            "selection_run_id": sel_id,
            "allow_simulated_dry_run": False,
            "framework_name": "pytest",
        },
    )
    assert exec_res.status_code == 200, f"Execution failed: {exec_res.text}"
    exec_data = exec_res.json()
    exec_id = exec_data["execution_run_id"]

    # Verify Real Docker Execution Properties
    assert exec_data["is_docker_isolated"] is True
    assert exec_data["docker_diagnostics"]["is_available"] is True
    assert exec_data["total_executed"] > 0
    assert len(exec_data["results"]) > 0

    for r in exec_data["results"]:
        assert r["duration_ms"] > 0
        assert r["exit_code"] in (0, 1), f"Test {r['test_name']} failed with exit code {r['exit_code']}: {r['error_message']}"
        assert len(r["stdout"]) > 0 or len(r["stderr"]) > 0

    # Verify Execution Status Transition in Selection Report
    updated_sel_report = client.get(f"/api/selection/{sel_id}").json()
    assert updated_sel_report["tests_executed_count"] > 0
    assert updated_sel_report["tests_passed_count"] > 0
    for tgt in updated_sel_report["targeted_generated_tests"]:
        if tgt["qualification_passed"]:
            assert tgt["execution_status"] in ("PASSED", "FAILED")

    # Verify Workflow Stage 3 State Transition
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.execution_run_id == exec_id
    assert wf.stages["execution"].status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS)

    # 6. Stage 4: Evidence-Based Failure Arbitration
    arb_res = client.post(
        "/api/arbitration/analyze",
        json={"workflow_id": wf_id, "execution_run_id": exec_id},
    )
    assert arb_res.status_code == 200, f"Arbitration failed: {arb_res.text}"
    arb_data = arb_res.json()
    arb_id = arb_data["arbitration_run_id"]

    assert arb_id.startswith("arb-")
    assert "category_counts" in arb_data
    assert "items" in arb_data

    # Verify Stage 4 Workflow Status
    wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert wf.arbitration_run_id == arb_id
    assert wf.stages["arbitration"].status == StageStatus.COMPLETED

    # 7. Verify Persistence and UI Reload Consistency
    reloaded_wf = workflow_storage.load_workflow(repo_id, wf_id)
    assert reloaded_wf.stages["evolution"].status == StageStatus.COMPLETED
    assert reloaded_wf.stages["impact"].status == StageStatus.COMPLETED
    assert reloaded_wf.stages["selection"].status == StageStatus.COMPLETED
    assert reloaded_wf.stages["execution"].status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS)
    assert reloaded_wf.stages["arbitration"].status == StageStatus.COMPLETED
