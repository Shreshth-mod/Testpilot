"""
Verification test suite for TestPilot Test Coverage Intelligence - Phase 2:
Function-Level Coverage and Uncovered Code Analysis.
Validates:
- Fully covered, partially covered, and completely uncovered functions.
- Class methods, static methods, and nested functions.
- Comments and blank lines do not distort coverage percentages.
- Files with no executable statements (e.g. empty or docstring only).
- Repository, file, class, and function summaries.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import pytest

from testpilot.coverage.collector import CoverageCollector
from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
from testpilot.coverage.models import (
    CoverageReport,
    CoverageStatus,
    FunctionStatus,
    LineCoverageData,
)
from testpilot.coverage.storage import CoverageStore


@pytest.fixture
def temp_phase2_repo():
    """
    Creates an isolated repository with a rich source file containing:
    1. Fully covered function: `add(a, b)`
    2. Partially covered function with branch: `evaluate_score(score, strict)`
    3. Completely uncovered function: `uncalled_cleanup(tag)`
    4. Class with methods: `Calculator.compute` (covered) and `Calculator.reset` (uncovered)
    5. Nested function: `outer_fn` with `inner_helper`
    6. Blank lines and comments throughout
    """
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_p2_")
    repo = Path(tmpdir)

    src_file = repo / "services.py"
    src_code = (
        "# Service module for Phase 2 Verification\n"
        "\n"
        "# 1. Fully covered function\n"
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    res = a + b\n"
        "    return res\n"
        "\n"
        "# 2. Partially covered function\n"
        "def evaluate_score(score, strict=False):\n"
        "    if strict:\n"
        "        # Strict mode branch - will NOT be executed in tests\n"
        "        return score * 0.5\n"
        "    else:\n"
        "        # Normal mode branch - WILL be executed\n"
        "        return score * 1.0\n"
        "\n"
        "# 3. Completely uncovered function\n"
        "def uncalled_cleanup(tag):\n"
        "    msg = f'cleanup: {tag}'\n"
        "    return msg\n"
        "\n"
        "# 4. Class with covered and uncovered methods\n"
        "class Calculator:\n"
        "    def compute(self, x):\n"
        "        return x * 2\n"
        "\n"
        "    def reset(self):\n"
        "        # Uncalled method\n"
        "        self.val = 0\n"
        "        return True\n"
        "\n"
        "# 5. Nested functions\n"
        "def outer_wrapper(val):\n"
        "    def inner_step(x):\n"
        "        return x + 1\n"
        "    return inner_step(val)\n"
    )
    src_file.write_text(src_code, encoding="utf-8")

    # Empty file with only comments / docstrings (0 executable statements)
    empty_file = repo / "empty_mod.py"
    empty_file.write_text("# Just comments\n\"\"\"Docstring only\"\"\"\n\n", encoding="utf-8")

    # Test file
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_services.py"
    test_code = (
        "import services\n\n"
        "def test_all():\n"
        "    assert services.add(2, 3) == 5\n"
        "    assert services.evaluate_score(100, strict=False) == 100.0\n"
        "    calc = services.Calculator()\n"
        "    assert calc.compute(5) == 10\n"
        "    assert services.outer_wrapper(10) == 11\n"
    )
    test_file.write_text(test_code, encoding="utf-8")

    yield repo
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_store():
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_p2_store_")
    db_path = Path(tmpdir) / "test_cov.db"
    reports_dir = Path(tmpdir) / "reports"
    store = CoverageStore(db_path=db_path, reports_dir=reports_dir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_function_level_categorization(temp_phase2_repo, temp_store):
    """Verify fully covered, partially covered, and uncovered functions are accurately classified."""
    collector = CoverageCollector(store=temp_store)

    report = collector.collect(
        repo_path=temp_phase2_repo,
        is_full_suite=True,
        repo_id="phase2_repo",
    )

    assert report.status == CoverageStatus.SUCCESS
    assert report.total_functions > 0
    assert len(report.file_summaries) > 0

    services_summary = next(f for f in report.file_summaries if "services.py" in f.file_path)
    func_map = {fn.qualified_name: fn for fn in services_summary.functions}

    # 1. `add` must be FULLY_COVERED (100%)
    assert "add" in func_map
    assert func_map["add"].status == FunctionStatus.FULLY_COVERED
    assert func_map["add"].coverage_pct == 100.0
    assert len(func_map["add"].missing_lines) == 0

    # 2. `evaluate_score` must be PARTIALLY_COVERED (>0% and <100%)
    assert "evaluate_score" in func_map
    assert func_map["evaluate_score"].status == FunctionStatus.PARTIALLY_COVERED
    assert 0.0 < func_map["evaluate_score"].coverage_pct < 100.0
    assert len(func_map["evaluate_score"].missing_lines) > 0

    # 3. `uncalled_cleanup` must be UNCOVERED (0%)
    assert "uncalled_cleanup" in func_map
    assert func_map["uncalled_cleanup"].status == FunctionStatus.UNCOVERED
    assert func_map["uncalled_cleanup"].coverage_pct == 0.0
    assert len(func_map["uncalled_cleanup"].executed_lines) == 0


def test_class_methods_and_nested_functions(temp_phase2_repo, temp_store):
    """Verify class methods and nested functions are appropriately resolved and measured."""
    collector = CoverageCollector(store=temp_store)

    report = collector.collect(
        repo_path=temp_phase2_repo,
        is_full_suite=True,
    )

    services_summary = next(f for f in report.file_summaries if "services.py" in f.file_path)
    func_map = {fn.qualified_name: fn for fn in services_summary.functions}

    # Calculator.compute should be FULLY_COVERED
    assert "Calculator.compute" in func_map
    assert func_map["Calculator.compute"].class_name == "Calculator"
    assert func_map["Calculator.compute"].status == FunctionStatus.FULLY_COVERED

    # Calculator.reset was not called in tests -> UNCOVERED
    assert "Calculator.reset" in func_map
    assert func_map["Calculator.reset"].class_name == "Calculator"
    assert func_map["Calculator.reset"].status == FunctionStatus.UNCOVERED

    # Nested function in outer_wrapper
    nested_fns = [fn for fn in services_summary.functions if fn.is_nested]
    assert len(nested_fns) >= 1
    assert any("inner_step" in fn.qualified_name for fn in nested_fns)


def test_comments_and_blank_lines_do_not_distort_coverage(temp_phase2_repo, temp_store):
    """Verify that comments, docstrings, and blank lines are excluded from executable statement counts."""
    collector = CoverageCollector(store=temp_store)

    report = collector.collect(
        repo_path=temp_phase2_repo,
        is_full_suite=True,
    )

    services_summary = next(f for f in report.file_summaries if "services.py" in f.file_path)
    add_fn = next(fn for fn in services_summary.functions if fn.function_name == "add")

    # In services.py, `add` has:
    # def add(a, b):  (line 4)
    #     """Docstring""" (line 5 - docstring)
    #     res = a + b  (line 6)
    #     return res   (line 7)
    # Only executable lines are 6, 7 (or 4, 6, 7 if def is recorded).
    # Line 5 (docstring) and blank lines must NOT be listed as missing!
    assert 5 not in add_fn.missing_lines
    for missing_line in add_fn.missing_lines:
        assert missing_line in add_fn.executable_lines


def test_file_with_no_executable_statements(temp_phase2_repo, temp_store):
    """Verify that a file with only comments/docstrings does not cause errors or false missing statements."""
    collector = CoverageCollector(store=temp_store)

    report = collector.collect(
        repo_path=temp_phase2_repo,
        is_full_suite=True,
    )

    empty_summary = next((f for f in report.file_summaries if "empty_mod.py" in f.file_path), None)
    if empty_summary:
        assert empty_summary.total_functions == 0
        assert empty_summary.missing_statements == 0


def test_uncovered_functions_query_from_store(temp_phase2_repo, temp_store):
    """Verify that CoverageStore.get_uncovered_functions returns accurate uncovered functions."""
    collector = CoverageCollector(store=temp_store)

    report = collector.collect(
        repo_path=temp_phase2_repo,
        is_full_suite=True,
    )

    # Query strictly UNCOVERED
    uncovered = temp_store.get_uncovered_functions(report.coverage_run_id, include_partial=False)
    uncovered_names = [u.function_name for u in uncovered]
    assert "uncalled_cleanup" in uncovered_names
    assert "reset" in uncovered_names
    assert "add" not in uncovered_names

    # Query UNCOVERED + PARTIAL
    with_partial = temp_store.get_uncovered_functions(report.coverage_run_id, include_partial=True)
    all_names = [u.function_name for u in with_partial]
    assert "evaluate_score" in all_names

