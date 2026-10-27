"""
Comprehensive Verification Suite for TestPilot Intelligent Test Generation Workflow.

Tests verify:
1. Every affected file is analyzed and classified into the 9 change categories.
2. Low-uncertainty files are not omitted from analysis.
3. Existing test selection is separated from new targeted generation.
4. Targeted tests are generated for uncovered behaviors (contracts, branches, boundaries, errors, integrations).
5. Non-behavioral (documentation-only) changes do NOT generate pointless tests.
6. Meaningful assertions exercise actual code; trivial/tautological assertions are rejected during qualification.
7. Generated tests are persisted in isolated workspace with execution_status='NOT_RUN' until executed.
8. Execution lifecycle transitions status only upon actual test run, preserving Docker isolation without host fallback.
9. Language scope adheres strictly to Python and JS/TS without Go support.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import pytest

from testpilot.core.workflow_context import (
    StageStatus,
    WorkflowContext,
    WorkflowStorageManager,
)
from testpilot.impact.models import (
    BoundaryImpact,
    ImpactAnalysisReport,
)
from testpilot.selection.models import (
    AffectedFileAnalysis,
    FileChangeClassification,
    SelectedTest,
    TargetedTest,
    TestSelectionReport,
)
from testpilot.selection.engine import TestSelectionEngine
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import TestCaseResult, TestOutcome, DockerDiagnostics
from testpilot.execution.runner import PytestDockerRunner, DockerUnavailableError
from testpilot.capabilities.detector import CapabilityDetector
from testpilot.capabilities.models import EcosystemType, PipelineStage, SupportLevel


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Creates an isolated workflow storage workspace."""
    ws_dir = tmp_path / "testpilot_runs"
    ws_dir.mkdir(parents=True, exist_ok=True)
    storage = WorkflowStorageManager(base_dir=ws_dir)
    return storage, ws_dir


def test_file_change_classification_all_categories(temp_workspace):
    """Verify all 9 change classifications are properly assigned with rationale and behaviors."""
    storage, ws_dir = temp_workspace
    engine = TestSelectionEngine(workflow_storage=storage)

    test_cases = [
        ("testbed/app/services/order_service.py", "def calculate_order_totals(): pass", FileChangeClassification.BUSINESS_LOGIC),
        ("testbed/app/routes/api.py", "from fastapi import APIRouter", FileChangeClassification.API_INTERFACE),
        ("testbed/app/models.py", "from pydantic import BaseModel\nclass Order(BaseModel): pass", FileChangeClassification.VALIDATION_BOUNDARY),
        ("testbed/app/errors.py", "class OrderException(Exception): pass", FileChangeClassification.ERROR_HANDLING),
        ("pyproject.toml", "[tool.poetry]", FileChangeClassification.CONFIGURATION),
        ("Dockerfile", "FROM python:3.11", FileChangeClassification.CONFIGURATION),
        ("testbed/app/client.py", "class PaymentGatewayClient:", FileChangeClassification.INTEGRATION),
        ("tests/test_order_service.py", "def test_order(): pass", FileChangeClassification.TESTS_ONLY),
        ("README.md", "# Project Documentation", FileChangeClassification.DOCUMENTATION_ONLY),
        ("docs/index.rst", "Sphinx documentation", FileChangeClassification.DOCUMENTATION_ONLY),
        ("scripts/deploy.sh", "#!/bin/bash", FileChangeClassification.OTHER),
    ]

    for f_path, content, expected_type in test_cases:
        ctype, reason, behavior = engine._classify_file_change(f_path, content)
        assert ctype == expected_type, f"Failed for {f_path}: expected {expected_type}, got {ctype}"
        assert len(reason) > 0
        assert len(behavior) > 0


def test_analyze_affected_files_does_not_omit_low_uncertainty_files(temp_workspace):
    """Verify analyze_affected_files identifies all changed files regardless of uncertainty."""
    storage, ws_dir = temp_workspace
    engine = TestSelectionEngine(workflow_storage=storage)

    # Create dummy workflow context
    wf = WorkflowContext(
        repo_id="test-repo",
        repo_path=".",
        base_commit_hash="base123",
        target_commit_hash="target456",
    )

    impact_report = ImpactAnalysisReport(
        workflow_id=wf.workflow_id,
        impact_run_id="imp-1",
        evolution_run_id="evo-1",
        repo_id=wf.repo_id,
        repo_path=".",
        base_commit_hash="base123",
        target_commit_hash="target456",
        changed_files=[
            "testbed/app/services/order_service.py",
            "testbed/app/models.py",
            "README.md",
            "pyproject.toml",
            "testbed/app/errors.py",
        ],
        direct_callers=[{"caller_file": "testbed/app/routes/api.py", "caller_symbol": "create_order"}],
        changed_symbols=[
            {"name": "calculate_order_totals", "file_path": "testbed/app/services/order_service.py"},
            {"name": "CartItem", "file_path": "testbed/app/models.py"},
            {"name": "OrderException", "file_path": "testbed/app/errors.py"},
        ],
    )

    selected_tests = [
        SelectedTest(
            test_name="test_subtotal_nominal",
            test_file="tests/test_order_service.py",
            targeted_symbol="calculate_order_totals",
            selection_tier="HIGH",
            selection_reason="Direct symbol coverage",
            selection_criteria=["CHANGED_CODE_COVERAGE"],
            execution_command="pytest tests/test_order_service.py -k test_subtotal_nominal",
        )
    ]

    affected = engine._analyze_affected_files(Path("."), impact_report, selected_tests, wf)

    # Verify every file is reported
    file_paths = [af.file_path for af in affected]
    assert "testbed/app/services/order_service.py" in file_paths
    assert "testbed/app/models.py" in file_paths
    assert "README.md" in file_paths
    assert "pyproject.toml" in file_paths
    assert "testbed/app/errors.py" in file_paths
    assert "testbed/app/routes/api.py" in file_paths

    # Verify order_service is covered by existing test
    af_order = next(af for af in affected if "order_service.py" in af.file_path)
    assert af_order.coverage_status == "covered_by_existing_tests"
    assert af_order.is_uncovered is False

    # Verify README.md is marked non-behavioral
    af_readme = next(af for af in affected if "README.md" in af.file_path)
    assert af_readme.change_type == FileChangeClassification.DOCUMENTATION_ONLY
    assert af_readme.coverage_status == "non_behavioral_no_test_needed"
    assert af_readme.is_uncovered is False

    # Verify models.py and errors.py are identified as needing tests
    af_models = next(af for af in affected if "models.py" in af.file_path)
    assert af_models.change_type == FileChangeClassification.VALIDATION_BOUNDARY
    assert af_models.coverage_status == "needs_generated_tests"
    assert af_models.is_uncovered is True

    af_errors = next(af for af in affected if "errors.py" in af.file_path)
    assert af_errors.change_type == FileChangeClassification.ERROR_HANDLING
    assert af_errors.coverage_status == "needs_generated_tests"
    assert af_errors.is_uncovered is True


def test_anti_trivial_assertion_qualification(temp_workspace):
    """Verify qualification rejects trivial/tautological assertions and accepts meaningful ones."""
    storage, _ = temp_workspace
    engine = TestSelectionEngine(workflow_storage=storage)

    # 1. Trivial literal comparison: assert "a" == "a"
    trivial_code_1 = """def test_trivial_1():
    assert "boundary" == "boundary"
"""
    _, passed_1, failures_1 = engine._qualify_test_candidate(trivial_code_1, "boundary")
    assert not passed_1
    assert any("trivial literal" in f for f in failures_1)

    # 2. Trivial len assertion: assert len("some_literal") > 0
    trivial_code_2 = """def test_trivial_2():
    boundary_expr = "unit_price >= 0"
    assert len(boundary_expr) > 0
"""
    _, passed_2, failures_2 = engine._qualify_test_candidate(trivial_code_2, "boundary")
    assert not passed_2
    assert any("trivial literal" in f for f in failures_2)

    # 3. Trivial boolean constant: assert True
    trivial_code_3 = """def test_trivial_3():
    assert True
"""
    _, passed_3, failures_3 = engine._qualify_test_candidate(trivial_code_3, "boundary")
    assert not passed_3

    # 4. Meaningful assertion: exercises model validation
    meaningful_code_1 = """def test_meaningful_model():
    from pydantic import ValidationError
    from testbed.app.models import CartItem
    with pytest.raises(ValidationError):
        CartItem(item_id="err", name="Err", unit_price=-1.0, quantity=1)
"""
    details_1, passed_m1, _ = engine._qualify_test_candidate(meaningful_code_1, "CartItem")
    assert passed_m1
    assert details_1["has_non_trivial_assertions"] is True
    assert details_1["collectable"] is True
    assert details_1["ast_syntax_valid"] is True

    # 5. Meaningful assertion: exercises function output and properties
    meaningful_code_2 = """def test_meaningful_service():
    from testbed.app.services.order_service import OrderService
    items = [{"item_id": "1", "name": "A", "unit_price": 50.0, "quantity": 1}]
    totals = OrderService.calculate_order_totals(items, None)
    assert totals is not None
    assert totals.subtotal == 50.0
"""
    details_2, passed_m2, _ = engine._qualify_test_candidate(meaningful_code_2, "OrderService")
    assert passed_m2
    assert details_2["has_non_trivial_assertions"] is True


def test_targeted_test_generation_uncovered_behavior_and_isolation(temp_workspace):
    """Verify targeted tests exercise actual code, are persisted in isolated workspace, and have NOT_RUN status."""
    storage, ws_dir = temp_workspace
    engine = TestSelectionEngine(workflow_storage=storage)

    wf = WorkflowContext(
        repo_id="testbed",
        repo_path=".",
        base_commit_hash="base123",
        target_commit_hash="target456",
    )
    storage.save_workflow(wf)

    impact_report = ImpactAnalysisReport(
        workflow_id=wf.workflow_id,
        impact_run_id="imp-2",
        evolution_run_id="evo-2",
        repo_id=wf.repo_id,
        repo_path=".",
        base_commit_hash="base123",
        target_commit_hash="target456",
        changed_symbols=[
            {"name": "calculate_order_totals", "file_path": "testbed/app/services/order_service.py"}
        ],
        boundary_impacts=[
            BoundaryImpact(
                target_symbol="CartItem.unit_price",
                target_file="testbed/app/models.py",
                boundary_type="LOWER_BOUND",
                constraint_expression="unit_price >= 0.0",
                source_spec="testbed/openapi.json",
                confidence=0.95,
            )
        ],
        changed_files=["testbed/app/services/order_service.py", "README.md"],
    )

    tests = engine.generate_targeted_tests(
        workflow_id=wf.workflow_id,
        impact_report=impact_report,
        max_tests=4,
    )

    assert len(tests) >= 2
    # Verify no test was generated for README.md
    assert all("README" not in t.target_file for t in tests)

    # Verify execution_status is NOT_RUN initially (never PASSED before execution)
    for t in tests:
        assert t.execution_status == "NOT_RUN"
        assert t.qualification_passed is True
        assert t.file_path.endswith(".py")
        assert (Path(".") / t.file_path).exists()
        # Verify non-trivial code
        assert "assert " in t.code or "with pytest.raises" in t.code


def test_execution_status_transition_and_metric_aggregation(temp_workspace):
    """Verify execution engine runs qualified tests and updates TargetedTest status to PASSED/FAILED."""
    storage, _ = temp_workspace
    sel_engine = TestSelectionEngine(workflow_storage=storage)
    exec_engine = IsolatedExecutionEngine(workflow_storage=storage, selection_engine=sel_engine)

    wf = WorkflowContext.create_initial(
        repo_id="testbed",
        repo_path=".",
        repo_name="testbed",
        base_commit_hash="base123",
        target_commit_hash="target456",
    )
    wf.stages["impact"].status = StageStatus.COMPLETED
    wf.stages["impact"].run_id = "imp-test"
    wf.impact_run_id = "imp-test"
    storage.save_workflow(wf)

    # Run selection
    sel_report = sel_engine.select(workflow_id=wf.workflow_id, generate_targeted=True, max_targeted_tests=3)
    assert len(sel_report.targeted_generated_tests) > 0
    assert sel_report.tests_executed_count == 0
    assert sel_report.tests_passed_count == 0

    # Execute tests using dry-run simulation to verify status mapping
    exec_report = exec_engine.execute(
        workflow_id=wf.workflow_id,
        allow_simulated_dry_run=True,
    )

    assert exec_report.total_executed > 0

    # Reload selection report to check execution status updates
    updated_sel_report = sel_engine.get_report(sel_report.selection_run_id)
    assert updated_sel_report is not None
    assert updated_sel_report.tests_executed_count == exec_report.total_executed

    # Verify that tested targets are either PASSED or FAILED (not falsely marked PASSED if not executed)
    for tgt in updated_sel_report.targeted_generated_tests:
        if tgt.test_name in [r.test_name for r in exec_report.results]:
            assert tgt.execution_status in ("PASSED", "FAILED")
        else:
            assert tgt.execution_status == "NOT_RUN"


def test_docker_isolation_rejects_unsafe_host_execution():
    """Verify that Docker runner never falls back to host execution when Docker is unavailable."""
    runner = PytestDockerRunner()

    # Simulate Docker unavailable
    class MockUnavailableRunner(PytestDockerRunner):
        def check_docker_availability(self):
            return DockerDiagnostics(
                is_available=False,
                daemon_version=None,
                error_message="Docker Desktop daemon is stopped.",
            )

    mock_runner = MockUnavailableRunner()

    with pytest.raises(DockerUnavailableError) as exc_info:
        mock_runner.run_tests(
            test_cases=[{"test_name": "test_sample", "test_file": "test.py"}],
            repo_path=Path("."),
            allow_simulated_dry_run=False,
        )

    assert "Docker daemon unavailable" in str(exc_info.value)
    assert "Unsafe local host execution is rejected" in str(exc_info.value)


def test_language_scope_python_and_js_no_go():
    """Verify that ecosystem capabilities maintain Python full workflow, JS/TS limited scope, and no Go support."""
    # 1. EcosystemType must only contain PYTHON, JAVASCRIPT, TYPESCRIPT, and UNKNOWN
    ecosystem_values = [e.value for e in EcosystemType]
    assert "python" in ecosystem_values
    assert "javascript" in ecosystem_values
    assert "typescript" in ecosystem_values
    assert "go" not in ecosystem_values
    assert "golang" not in ecosystem_values

    # 2. Python repository gets full capability assessment
    report = CapabilityDetector.generate_report(Path("."))
    assert report.ecosystem.primary_language.lower() in ("python", "unknown")

    # 3. JS/TS capabilities do NOT falsely claim test generation or Docker execution
    from testpilot.capabilities.models import RepositoryEcosystem
    js_eco = RepositoryEcosystem(
        primary_language="JavaScript",
        detected_languages=["JavaScript"],
    )
    caps = CapabilityDetector.assess_stage_capabilities(
        repo_path=Path("."),
        ecosystem=js_eco,
        is_git=True,
    )
    assert caps[PipelineStage.TEST_GENERATION.value].support_level == SupportLevel.UNSUPPORTED
    assert caps[PipelineStage.DOCKER_EXECUTION.value].support_level == SupportLevel.UNSUPPORTED
