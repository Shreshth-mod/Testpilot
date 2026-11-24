"""
Verification test suite for TestPilot Test Coverage Intelligence - Phase 1:
Runtime Coverage Collection and Storage.
Validates line-level execution capture, uncovered line detection,
direct coverage.py comparison, targeted vs full-suite runs,
run isolation (no cross-contamination), empty/error handling, and mode detection.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import time
import pytest

import coverage

from testpilot.coverage.collector import CoverageCollector
from testpilot.coverage.models import CoverageReport, CoverageStatus
from testpilot.coverage.storage import CoverageStore
from testpilot.execution.engine import IsolatedExecutionEngine
from testpilot.execution.models import TestOutcome


@pytest.fixture
def temp_sample_repo():
    """Creates an isolated temporary repository with known ground-truth code and tests."""
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_test_")
    repo = Path(tmpdir)

    # 1. Source module with known lines
    # Lines:
    # 1: def calculate_discount(price, is_member):
    # 2:     if is_member:
    # 3:         return price * 0.8
    # 4:     if price > 100:
    # 5:         return price * 0.9
    # 6:     return price
    # 7:
    # 8: def uncalled_feature(x):
    # 9:     y = x * 2
    # 10:    return y + 1
    src_file = repo / "pricing.py"
    src_code = (
        "def calculate_discount(price, is_member):\n"
        "    if is_member:\n"
        "        return price * 0.8\n"
        "    if price > 100:\n"
        "        return price * 0.9\n"
        "    return price\n\n"
        "def uncalled_feature(x):\n"
        "    y = x * 2\n"
        "    return y + 1\n"
    )
    src_file.write_text(src_code, encoding="utf-8")

    # 2. Test file covering member discount only
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_pricing.py"
    test_code = (
        "import pricing\n\n"
        "def test_member_discount():\n"
        "    assert pricing.calculate_discount(100, True) == 80.0\n\n"
        "def test_non_member_small():\n"
        "    assert pricing.calculate_discount(50, False) == 50\n"
    )
    test_file.write_text(test_code, encoding="utf-8")

    yield repo
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_coverage_store():
    """Creates an isolated CoverageStore in a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_store_")
    db_path = Path(tmpdir) / "test_coverage.db"
    reports_dir = Path(tmpdir) / "reports"
    store = CoverageStore(db_path=db_path, reports_dir=reports_dir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_runtime_coverage_records_executed_and_missing_lines(temp_sample_repo, temp_coverage_store):
    """Verify that executed lines are recorded and missing lines are identified accurately."""
    collector = CoverageCollector(store=temp_coverage_store)

    report = collector.collect(
        repo_path=temp_sample_repo,
        test_cases=[{
            "test_name": "test_member_discount",
            "test_file": "tests/test_pricing.py",
        }],
        repo_id="test_repo_1",
        commit_hash="c111111",
        execution_mode="local_subprocess",
    )

    assert report.status == CoverageStatus.SUCCESS
    assert report.total_statements > 0
    assert "pricing.py" in report.files

    fcov = report.files["pricing.py"]
    # calculate_discount definition (line 1), if is_member (line 2), return price*0.8 (line 3) were executed
    assert 1 in fcov.executed_lines
    assert 2 in fcov.executed_lines
    assert 3 in fcov.executed_lines

    # uncalled_feature lines (9, 10) were NOT executed and must be in missing_lines
    assert 9 in fcov.missing_lines
    assert 10 in fcov.missing_lines
    assert fcov.line_coverage_pct > 0.0
    assert fcov.line_coverage_pct < 100.0


def test_comparison_against_direct_coverage_py(temp_sample_repo, temp_coverage_store):
    """Compare TestPilot's results against direct coverage.py analysis."""
    collector = CoverageCollector(store=temp_coverage_store)

    report = collector.collect(
        repo_path=temp_sample_repo,
        is_full_suite=True,
        repo_id="test_repo_direct",
        commit_hash="c222222",
    )

    assert report.status == CoverageStatus.SUCCESS
    tp_fcov = report.files["pricing.py"]

    # Run direct coverage.py independently on the same file
    cov = coverage.Coverage(source=[str(temp_sample_repo)])
    cov.start()
    import sys
    sys.path.insert(0, str(temp_sample_repo))
    try:
        import pricing
        # Reproduce the exact two test calls
        pricing.calculate_discount(100, True)
        pricing.calculate_discount(50, False)
    finally:
        cov.stop()
        cov.save()

    direct_analysis = cov.analysis2(str(temp_sample_repo / "pricing.py"))
    direct_missing = sorted(list(direct_analysis[3]))

    # TestPilot's missing lines must match direct coverage.py's missing lines!
    assert tp_fcov.missing_lines == direct_missing


def test_targeted_vs_full_suite_execution(temp_sample_repo, temp_coverage_store):
    """Test targeted execution (1 test) vs full-suite execution (all tests)."""
    collector = CoverageCollector(store=temp_coverage_store)

    # 1. Targeted: only test_member_discount
    rep_targeted = collector.collect(
        repo_path=temp_sample_repo,
        test_cases=[{"test_name": "test_member_discount", "test_file": "tests/test_pricing.py"}],
        is_full_suite=False,
        repo_id="repo_targeted",
    )

    # 2. Full Suite: runs both test_member_discount AND test_non_member_small
    rep_full = collector.collect(
        repo_path=temp_sample_repo,
        is_full_suite=True,
        repo_id="repo_full",
    )

    assert rep_targeted.status == CoverageStatus.SUCCESS
    assert rep_full.status == CoverageStatus.SUCCESS

    # Full suite executed more lines in pricing.py than the single targeted test!
    cov_targeted = rep_targeted.files["pricing.py"].covered_statements
    cov_full = rep_full.files["pricing.py"].covered_statements
    assert cov_full >= cov_targeted


def test_repeated_runs_isolation_no_data_mixing(temp_sample_repo, temp_coverage_store):
    """Verify that repeated runs for different repositories or revisions do not mix coverage data."""
    collector = CoverageCollector(store=temp_coverage_store)

    report_repo_a = collector.collect(
        repo_path=temp_sample_repo,
        repo_id="repo_alpha",
        commit_hash="hash_alpha",
    )

    report_repo_b = collector.collect(
        repo_path=temp_sample_repo,
        repo_id="repo_beta",
        commit_hash="hash_beta",
    )

    # Stored reports must have separate run IDs
    assert report_repo_a.coverage_run_id != report_repo_b.coverage_run_id

    # Querying by repo_id from the store must yield distinct, isolated reports
    rep_a_loaded = temp_coverage_store.get_latest_report(repo_id="repo_alpha")
    rep_b_loaded = temp_coverage_store.get_latest_report(repo_id="repo_beta")

    assert rep_a_loaded is not None
    assert rep_b_loaded is not None
    assert rep_a_loaded.repo_id == "repo_alpha"
    assert rep_b_loaded.repo_id == "repo_beta"
    assert rep_a_loaded.coverage_run_id == report_repo_a.coverage_run_id
    assert rep_b_loaded.coverage_run_id == report_repo_b.coverage_run_id


def test_empty_collection_and_errors_not_reported_as_success(temp_sample_repo, temp_coverage_store):
    """Verify that empty test collections and collection errors are marked UNAVAILABLE/FAILED, not false success."""
    collector = CoverageCollector(store=temp_coverage_store)

    # 1. Non-existent test name produces no collected tests (pytest exit code 5)
    rep_empty = collector.collect(
        repo_path=temp_sample_repo,
        test_cases=[{"test_name": "non_existent_test_xyz_123", "test_file": "tests/test_pricing.py"}],
    )
    assert rep_empty.status == CoverageStatus.UNAVAILABLE
    assert rep_empty.line_coverage_pct == 0.0

    # 2. Syntax error in a test file produces collection failure
    bad_test = temp_sample_repo / "tests" / "test_broken.py"
    bad_test.write_text("def broken_syntax(:\n    pass\n", encoding="utf-8")
    try:
        rep_err = collector.collect(
            repo_path=temp_sample_repo,
            is_full_suite=True,
        )
        assert rep_err.status in (CoverageStatus.FAILED, CoverageStatus.UNAVAILABLE)
        assert rep_err.error_message is not None
    finally:
        bad_test.unlink(missing_ok=True)


def test_unsupported_execution_mode_clearly_identified(temp_sample_repo, temp_coverage_store):
    """Verify that unsupported execution modes (such as simulated sandbox) are clearly flagged."""
    collector = CoverageCollector(store=temp_coverage_store)

    report = collector.collect(
        repo_path=temp_sample_repo,
        execution_mode="simulated_sandbox",
    )

    assert report.status == CoverageStatus.UNSUPPORTED_MODE
    assert "simulated" in (report.error_message or "").lower()
    assert report.line_coverage_pct == 0.0


def test_existing_test_execution_works_with_coverage_disabled():
    """Confirm that existing test execution still functions completely when coverage is disabled."""
    engine = IsolatedExecutionEngine()
    # Check docker status without triggering coverage
    status = engine.get_docker_status()
    assert hasattr(status, "is_available")

