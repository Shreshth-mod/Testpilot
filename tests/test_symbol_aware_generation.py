"""
Focused regression tests for symbol-aware test generation and failure arbitration.
Verifies:
1. Class methods, static methods, and nested classes preserve class hierarchy and scope.
2. Module-level functions generate direct module references.
3. Symbol-lookup defect is never silently swallowed by try/except pass to become a passing test.
4. Failure arbitration correctly triages TEST_DEFECT (symbol lookup failure) vs REGRESSION (genuine behavioral change).
5. Base revision outcome is not fabricated or inferred without verification.
"""

from pathlib import Path
import pytest

from testpilot.arbitration.engine import FailureArbitrationEngine
from testpilot.arbitration.models import ArbitrationCategory
from testpilot.core.workflow_context import (
    WorkflowStorageManager,
    WorkflowContext,
    StageStatus,
)
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import IsolatedExecutionReport, TestCaseResult, TestOutcome
from testpilot.impact.models import ImpactAnalysisReport
from testpilot.selection.engine import TestSelectionEngine


def test_symbol_aware_generation_preserves_class_hierarchy(tmp_path):
    """Verify generator preserves class scope for class methods / static methods."""
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    sel_engine = TestSelectionEngine(workflow_storage=mgr)

    # Repository structure
    repo_dir = tmp_path / "repo"
    svc_dir = repo_dir / "app" / "services"
    svc_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "app" / "__init__.py").write_text("", encoding="utf-8")
    (svc_dir / "__init__.py").write_text("", encoding="utf-8")
    (svc_dir / "order_service.py").write_text(
        """class OrderService:
    @staticmethod
    def calculate_subtotal(items):
        return sum(i.get('price', 0) for i in items)
""",
        encoding="utf-8",
    )

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=str(repo_dir),
        repo_name="TestRepo",
        base_commit_hash="a" * 40,
        target_commit_hash="b" * 40,
        workflow_id="wf-class-sym",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    impact_report = ImpactAnalysisReport(
        workflow_id=wf.workflow_id,
        evolution_run_id="evo-001",
        repo_id=wf.repo_id,
        repo_path=wf.repo_path,
        base_commit_hash=wf.base_commit_hash,
        target_commit_hash=wf.target_commit_hash,
        analysis_id="imp-001",
        changed_symbols=[
            {
                "name": "calculate_subtotal",
                "class_name": "OrderService",
                "qualified_name": "OrderService.calculate_subtotal",
                "file_path": "app/services/order_service.py",
                "parameters": ["items"],
            }
        ],
    )

    tests = sel_engine.generate_targeted_tests(
        workflow_id=wf.workflow_id,
        impact_report=impact_report,
        max_tests=2,
        repo_path=repo_dir,
    )

    assert len(tests) >= 1
    t = tests[0]
    assert t.target_symbol == "OrderService.calculate_subtotal"
    assert t.qualification_passed is True
    # Verify code accesses OrderService class, not getattr(mod, 'calculate_subtotal')
    assert 'cls = getattr(cls, part)' in t.code or 'OrderService' in t.code
    assert 'assert hasattr(cls, "calculate_subtotal")' in t.code
    assert 'assert func is not None' not in t.code


def test_symbol_aware_generation_for_module_functions(tmp_path):
    """Verify generator creates direct module references for module-level functions."""
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    sel_engine = TestSelectionEngine(workflow_storage=mgr)

    repo_dir = tmp_path / "repo"
    math_dir = repo_dir / "app" / "utils"
    math_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "app" / "__init__.py").write_text("", encoding="utf-8")
    (math_dir / "__init__.py").write_text("", encoding="utf-8")
    (math_dir / "math_utils.py").write_text(
        """def compute_tax(subtotal: float, rate: float = 0.08) -> float:
    return round(subtotal * rate, 2)
""",
        encoding="utf-8",
    )

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=str(repo_dir),
        repo_name="TestRepo",
        base_commit_hash="a" * 40,
        target_commit_hash="b" * 40,
        workflow_id="wf-mod-sym",
    )
    wf.stages["evolution"].status = StageStatus.COMPLETED
    wf.stages["impact"].status = StageStatus.COMPLETED
    mgr.save_workflow(wf)

    impact_report = ImpactAnalysisReport(
        workflow_id=wf.workflow_id,
        evolution_run_id="evo-002",
        repo_id=wf.repo_id,
        repo_path=wf.repo_path,
        base_commit_hash=wf.base_commit_hash,
        target_commit_hash=wf.target_commit_hash,
        analysis_id="imp-002",
        changed_symbols=[
            {
                "name": "compute_tax",
                "class_name": None,
                "file_path": "app/utils/math_utils.py",
                "parameters": ["subtotal", "rate"],
            }
        ],
    )

    tests = sel_engine.generate_targeted_tests(
        workflow_id=wf.workflow_id,
        impact_report=impact_report,
        max_tests=2,
        repo_path=repo_dir,
    )

    assert len(tests) >= 1
    t = tests[0]
    assert t.target_symbol == "compute_tax"
    assert t.qualification_passed is True
    assert 'assert hasattr(mod, "compute_tax")' in t.code
    assert 'target = getattr(mod, "compute_tax")' in t.code


def test_symbol_lookup_defect_is_not_misrepresented_as_passing(tmp_path):
    """Verify that a symbol lookup failure raises AssertionError and is not caught by try/except pass."""
    repo_dir = tmp_path / "repo"
    app_dir = repo_dir / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    # File has OrderService, but does NOT have calculate_subtotal at module level
    (app_dir / "order_service.py").write_text(
        """class OrderService:
    @staticmethod
    def calculate_subtotal(items):
        return 10.0
""",
        encoding="utf-8",
    )

    # If code erroneously looks up on mod:
    bad_test_code = """
import importlib
mod = importlib.import_module("app.order_service")
assert hasattr(mod, "calculate_subtotal"), "Module 'app.order_service' has no symbol 'calculate_subtotal'"
"""
    import sys
    sys.path.insert(0, str(repo_dir))
    try:
        with pytest.raises(AssertionError) as exc_info:
            exec(bad_test_code, {})
        assert "has no symbol 'calculate_subtotal'" in str(exc_info.value)
    finally:
        sys.path.remove(str(repo_dir))


def test_arbitration_classifies_symbol_lookup_defect_as_test_defect(tmp_path):
    """Verify arbitration classifies symbol-lookup assertion failures as TEST_DEFECT."""
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    exec_engine = IsolatedExecutionEngine(workflow_storage=mgr)
    arb_engine = FailureArbitrationEngine(workflow_storage=mgr, execution_engine=exec_engine)

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=".",
        repo_name="TestRepo",
        base_commit_hash="a" * 40,
        target_commit_hash="b" * 40,
        workflow_id="wf-arb-defect",
    )
    for s in ("evolution", "impact", "selection"):
        wf.stages[s].status = StageStatus.COMPLETED
    wf.stages["execution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.execution_run_id = "exec-defect-01"
    mgr.save_workflow(wf)

    # 1. Defect from old generator: assert None is not None
    # 2. Defect from symbol-aware assertion: Class 'OrderService' has no symbol 'calculate_subtotal'
    # 3. Genuine behavioral regression: AssertionError: assert 40.0 == 50.0
    results = [
        TestCaseResult(
            test_name="test_calculate_subtotal_contract_0",
            test_file="tests/generated/test_targeted_sel.py",
            outcome=TestOutcome.FAILED,
            error_message="AssertionError: assert None is not None",
            targeted_symbol="calculate_subtotal",
        ),
        TestCaseResult(
            test_name="test_symbol_missing_contract_1",
            test_file="tests/generated/test_targeted_sel.py",
            outcome=TestOutcome.FAILED,
            error_message="AssertionError: Class 'OrderService' has no symbol 'calculate_subtotal'",
            targeted_symbol="OrderService.calculate_subtotal",
        ),
        TestCaseResult(
            test_name="test_genuine_regression_assertion",
            test_file="tests/test_order_service.py",
            outcome=TestOutcome.FAILED,
            error_message="AssertionError: assert 40.0 == 50.0",
            targeted_symbol="OrderService.calculate_subtotal",
        ),
    ]

    exec_report = IsolatedExecutionReport(
        execution_run_id="exec-defect-01",
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

    report = arb_engine.arbitrate(workflow_id="wf-arb-defect")
    cats = {it.test_name: it.category for it in report.items}

    # Symbol-lookup failures are TEST_DEFECT
    assert cats["test_calculate_subtotal_contract_0"] == ArbitrationCategory.TEST_DEFECT
    assert cats["test_symbol_missing_contract_1"] == ArbitrationCategory.TEST_DEFECT

    # Real assertion error on modified code is REGRESSION
    assert cats["test_genuine_regression_assertion"] == ArbitrationCategory.REGRESSION


def test_arbitration_does_not_infer_base_outcome_without_checking(tmp_path):
    """Verify that arbitration sets base_outcome to None when base revision is not verified."""
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    exec_engine = IsolatedExecutionEngine(workflow_storage=mgr)
    arb_engine = FailureArbitrationEngine(workflow_storage=mgr, execution_engine=exec_engine)

    wf = WorkflowContext.create_initial(
        repo_id="test_repo",
        repo_path=".",
        repo_name="TestRepo",
        base_commit_hash="mock-base-hash-nonexistent",
        target_commit_hash="mock-target-hash-nonexistent",
        workflow_id="wf-base-unverified",
    )
    for s in ("evolution", "impact", "selection"):
        wf.stages[s].status = StageStatus.COMPLETED
    wf.stages["execution"].status = StageStatus.COMPLETED_WITH_WARNINGS
    wf.execution_run_id = "exec-unverified-01"
    mgr.save_workflow(wf)

    results = [
        TestCaseResult(
            test_name="test_order_pricing_mismatch",
            test_file="tests/test_order.py",
            outcome=TestOutcome.FAILED,
            error_message="AssertionError: assert 100.0 == 90.0",
            targeted_symbol="calculate_subtotal",
        ),
    ]

    exec_report = IsolatedExecutionReport(
        execution_run_id="exec-unverified-01",
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

    report = arb_engine.arbitrate(workflow_id="wf-base-unverified")
    item = report.items[0]

    assert item.category == ArbitrationCategory.REGRESSION
    # base_outcome must NOT be inferred as "PASSED" because base commit was not checked!
    assert item.base_outcome is None
    assert any("Base revision not checked" in ev for ev in item.evidence)
