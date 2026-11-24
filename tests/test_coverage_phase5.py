"""
Phase 5 Verification Test Suite: Git Revision Comparison and AI Coverage Suggestions.
Validates:
1. Revision coverage comparison across Git commits/runs.
2. Accurate detection of coverage regressions (functions losing coverage).
3. Identification of newly covered functions and untested added/modified code.
4. Graceful handling of missing base coverage reports.
5. AI coverage suggestion analysis grounded in AST branches and boundary conditions.
6. Synthesizing runnable pytest code for uncovered code.
7. Verification that coverage improvement is only claimed when empirically confirmed by a subsequent run.
8. API endpoints for revision comparison and AI suggestions.
"""

import ast
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from testpilot.coverage.diff_coverage import RevisionCoverageComparator
from testpilot.coverage.models import (
    CoverageReport,
    CoverageStatus,
    FileCoverageSummary,
    FunctionCoverage,
    FunctionCoverageSuggestion,
    FunctionStatus,
    LineCoverageData,
    RevisionComparisonReport,
    SuggestionVerificationResult,
)
from testpilot.coverage.storage import CoverageStore
from testpilot.coverage.suggestions import CoverageSuggestionEngine
from testpilot.web.api import app


@pytest.fixture
def temp_store(tmp_path):
    db_file = tmp_path / "test_cov.db"
    rep_dir = tmp_path / "reports"
    return CoverageStore(db_path=db_file, reports_dir=rep_dir)


def test_revision_comparison_with_base_and_target(temp_store):
    """Verifies that comparator identifies regressions, newly covered, and untested functions."""
    comparator = RevisionCoverageComparator(store=temp_store)

    # Base report: commit_A
    base_fn_foo = FunctionCoverage(
        function_name="calculate_discount",
        qualified_name="OrderService.calculate_discount",
        file_path="app/order.py",
        start_line=10,
        end_line=25,
        executed_lines=[10, 12, 14, 16, 18],
        executable_lines=[10, 12, 14, 16, 18],
        coverage_pct=100.0,
        status=FunctionStatus.FULLY_COVERED,
    )
    base_fn_bar = FunctionCoverage(
        function_name="process_refund",
        qualified_name="OrderService.process_refund",
        file_path="app/order.py",
        start_line=30,
        end_line=45,
        missing_lines=[30, 32, 34],
        executable_lines=[30, 32, 34],
        coverage_pct=0.0,
        status=FunctionStatus.UNCOVERED,
    )
    base_report = CoverageReport(
        coverage_run_id="cov-base-001",
        repo_id="test-repo",
        commit_hash="commit_aaa111",
        total_statements=50,
        covered_statements=35,
        missing_statements=15,
        line_coverage_pct=70.0,
        total_functions=2,
        covered_functions=1,
        partially_covered_functions=0,
        uncovered_functions=1,
        function_coverage_pct=50.0,
        files={
            "app/order.py": LineCoverageData(
                file_path="app/order.py",
                total_statements=50,
                covered_statements=35,
                missing_lines=[30, 32, 34],
                executed_lines=[10, 12, 14, 16, 18],
                line_coverage_pct=70.0,
            )
        },
        file_summaries=[
            FileCoverageSummary(
                file_path="app/order.py",
                total_statements=50,
                covered_statements=35,
                line_coverage_pct=70.0,
                total_functions=2,
                covered_functions=1,
                uncovered_functions=1,
                function_coverage_pct=50.0,
                functions=[base_fn_foo, base_fn_bar],
            )
        ],
    )

    # Target report: commit_B
    # foo regressed to 40% coverage!
    # bar improved to 100% coverage!
    # baz is newly added and has 0% coverage!
    target_fn_foo = FunctionCoverage(
        function_name="calculate_discount",
        qualified_name="OrderService.calculate_discount",
        file_path="app/order.py",
        start_line=10,
        end_line=25,
        executed_lines=[10, 12],
        missing_lines=[14, 16, 18],
        executable_lines=[10, 12, 14, 16, 18],
        coverage_pct=40.0,
        status=FunctionStatus.PARTIALLY_COVERED,
    )
    target_fn_bar = FunctionCoverage(
        function_name="process_refund",
        qualified_name="OrderService.process_refund",
        file_path="app/order.py",
        start_line=30,
        end_line=45,
        executed_lines=[30, 32, 34],
        executable_lines=[30, 32, 34],
        coverage_pct=100.0,
        status=FunctionStatus.FULLY_COVERED,
    )
    target_fn_baz = FunctionCoverage(
        function_name="apply_vip_perks",
        qualified_name="OrderService.apply_vip_perks",
        file_path="app/order.py",
        start_line=50,
        end_line=65,
        missing_lines=[50, 52, 54],
        executable_lines=[50, 52, 54],
        coverage_pct=0.0,
        status=FunctionStatus.UNCOVERED,
    )
    target_report = CoverageReport(
        coverage_run_id="cov-target-002",
        repo_id="test-repo",
        commit_hash="commit_bbb222",
        total_statements=60,
        covered_statements=45,
        missing_statements=15,
        line_coverage_pct=75.0,
        total_functions=3,
        covered_functions=1,
        partially_covered_functions=1,
        uncovered_functions=1,
        function_coverage_pct=33.3,
        files={
            "app/order.py": LineCoverageData(
                file_path="app/order.py",
                total_statements=60,
                covered_statements=45,
                missing_lines=[14, 16, 18, 50, 52, 54],
                executed_lines=[10, 12, 30, 32, 34],
                line_coverage_pct=75.0,
            )
        },
        file_summaries=[
            FileCoverageSummary(
                file_path="app/order.py",
                total_statements=60,
                covered_statements=45,
                line_coverage_pct=75.0,
                total_functions=3,
                covered_functions=1,
                partially_covered_functions=1,
                uncovered_functions=1,
                function_coverage_pct=33.3,
                functions=[target_fn_foo, target_fn_bar, target_fn_baz],
            )
        ],
    )

    result = comparator.compare_reports(base_report, target_report)

    assert result.base_available is True
    assert result.base_commit == "commit_aaa111"
    assert result.target_commit == "commit_bbb222"
    assert result.line_coverage_delta == 5.0  # 75 - 70

    # 1. Regression detection
    assert len(result.regressions) == 1
    reg = result.regressions[0]
    assert reg.function_name == "OrderService.calculate_discount"
    assert reg.previous_coverage_pct == 100.0
    assert reg.current_coverage_pct == 40.0
    assert 14 in reg.lines_lost

    # 2. Newly covered functions
    assert any("process_refund" in f for f in result.newly_covered_functions)

    # 3. Untested new function
    assert any("apply_vip_perks" in f for f in result.uncovered_new_or_modified_functions)

    # 4. File-level diff
    assert len(result.files) == 1
    f_diff = result.files[0]
    assert f_diff.file_path == "app/order.py"
    assert f_diff.line_coverage_delta == 5.0


def test_revision_comparison_when_base_unavailable(temp_store):
    """Verifies graceful handling when base revision coverage is missing."""
    comparator = RevisionCoverageComparator(store=temp_store)

    target_fn = FunctionCoverage(
        function_name="dispatch_event",
        qualified_name="dispatch_event",
        file_path="events.py",
        start_line=1,
        end_line=10,
        coverage_pct=0.0,
        status=FunctionStatus.UNCOVERED,
    )
    target_report = CoverageReport(
        coverage_run_id="cov-only-target",
        repo_id="test-repo",
        commit_hash="commit_tgt999",
        total_statements=20,
        covered_statements=10,
        missing_statements=10,
        line_coverage_pct=50.0,
        total_functions=1,
        covered_functions=0,
        uncovered_functions=1,
        function_coverage_pct=0.0,
        files={
            "events.py": LineCoverageData(file_path="events.py", line_coverage_pct=50.0)
        },
        file_summaries=[
            FileCoverageSummary(
                file_path="events.py",
                total_statements=20,
                covered_statements=10,
                line_coverage_pct=50.0,
                total_functions=1,
                uncovered_functions=1,
                functions=[target_fn],
            )
        ],
    )

    result = comparator.compare_reports(None, target_report)

    assert result.base_available is False
    assert result.base_commit == "none"
    assert result.target_commit == "commit_tgt999"
    assert result.target_line_coverage_pct == 50.0
    assert len(result.notes) >= 1
    assert any("unavailable" in n.lower() for n in result.notes)
    assert any("dispatch_event" in f for f in result.uncovered_new_or_modified_functions)


def test_coverage_suggestion_engine_ast_analysis(tmp_path, temp_store):
    """Verifies that CoverageSuggestionEngine parses uncovered code and produces grounded scenarios."""
    # Create sample module with branch and exception
    sample_code = textwrap_dedent("""
        def calculate_shipping(weight: float, express: bool = False) -> float:
            if weight <= 0:
                raise ValueError("Weight must be positive")
            if express:
                return 15.00 + weight * 2.5
            return 5.00 + weight * 1.0
    """)
    sample_file = tmp_path / "shipping.py"
    sample_file.write_text(sample_code.strip(), encoding="utf-8")

    func_cov = FunctionCoverage(
        function_name="calculate_shipping",
        qualified_name="shipping.calculate_shipping",
        file_path="shipping.py",
        start_line=1,
        end_line=6,
        missing_lines=[2, 3, 4, 5, 6],
        executable_lines=[1, 2, 3, 4, 5, 6],
        coverage_pct=0.0,
        status=FunctionStatus.UNCOVERED,
    )

    suggester = CoverageSuggestionEngine(store=temp_store)
    suggestion = suggester.analyze_function(func_cov, repo_path=tmp_path)

    assert suggestion.function_name == "calculate_shipping"
    assert suggestion.current_status == FunctionStatus.UNCOVERED
    assert len(suggestion.scenarios) >= 2

    # Check scenarios
    titles = [s.title for s in suggestion.scenarios]
    assert any("primary flow" in t.lower() for t in titles)
    assert any("branch" in t.lower() or "exception" in t.lower() for t in titles)

    # Check synthesized test code syntax
    assert suggestion.synthesized_test_code is not None
    parsed = ast.parse(suggestion.synthesized_test_code)
    test_func_names = [n.name for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef)]
    assert len(test_func_names) >= 1
    assert any("calculate_shipping" in n for n in test_func_names)


def test_suggestion_verification_claims_improvement_only_when_empirically_measured(tmp_path, temp_store):
    """
    Verifies that suggestion verification only claims improved=True when coverage
    actually rises after running the generated tests.
    """
    suggester = CoverageSuggestionEngine(store=temp_store)

    suggestion = FunctionCoverageSuggestion(
        function_name="dummy_calc",
        qualified_name="dummy.dummy_calc",
        file_path="dummy.py",
        current_coverage_pct=0.0,
        current_status=FunctionStatus.UNCOVERED,
        missing_lines=[1, 2, 3],
        synthesized_test_code="def test_fake():\n    assert 1 == 1\n",
    )

    # When test execution does not touch dummy.py, coverage remains 0.0%
    # Verification MUST report coverage_improved=False
    res = suggester.verify_suggestion(suggestion, repo_path=tmp_path)

    assert res.target_function == "dummy.dummy_calc"
    assert res.coverage_before_pct == 0.0
    # Because dummy.py was not executed/covered, after pct is 0.0, so coverage_improved MUST be False
    assert res.coverage_improved is False
    assert res.verified is False
    assert "no improvement" in res.details.lower() or "failed" in res.details.lower() or res.coverage_after_pct == 0.0


def test_phase5_api_endpoints():
    """Verifies FastAPI endpoints for revision comparison and AI suggestions."""
    client = TestClient(app)

    # 1. GET /api/coverage/compare
    resp = client.get("/api/coverage/compare?base_commit=HEAD~1&target_commit=HEAD")
    assert resp.status_code == 200
    data = resp.json()
    assert "base_available" in data
    assert "line_coverage_delta" in data
    assert "regressions" in data
    assert "newly_covered_functions" in data

    # 2. GET /api/coverage/suggestions
    resp2 = client.get("/api/coverage/suggestions")
    assert resp2.status_code == 200
    assert isinstance(resp2.json(), list)

    # 3. POST /api/coverage/suggestions/generate
    resp3 = client.post(
        "/api/coverage/suggestions/generate",
        json={
            "function_name": "example_fn",
            "file_path": "nonexistent.py",
            "repo_path": ".",
            "custom_code": "def test_noop(): pass\n",
        },
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert "coverage_improved" in data3
    assert "coverage_delta_pct" in data3
    assert "verified" in data3


def textwrap_dedent(text: str) -> str:
    import textwrap
    return textwrap.dedent(text)

