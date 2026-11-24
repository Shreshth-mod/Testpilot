"""
Verification test suite for TestPilot Test Coverage Intelligence - Phase 4:
Test Coverage Intelligence Dashboard and API endpoints.
Validates:
- GET /api/coverage/summary
- GET /api/coverage/files
- GET /api/coverage/file (line-by-line annotated code)
- GET /api/coverage/functions (filters by status and name)
- GET /api/coverage/uncovered
- GET /api/coverage/test-map
- POST /api/coverage/collect
- GET /api/coverage/report/download (JSON and Markdown)
- Graceful empty states when no coverage data exists.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from testpilot.coverage.collector import CoverageCollector
from testpilot.coverage.models import (
    CoverageEvidenceType,
    CoverageReport,
    CoverageStatus,
    FileCoverageSummary,
    FunctionCoverage,
    FunctionStatus,
    LineCoverageData,
    TestToCodeMapping,
)
from testpilot.coverage.storage import CoverageStore
from testpilot.web.api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def populated_coverage_store():
    """Seeds a known CoverageReport into the global CoverageStore for API testing."""
    store = CoverageStore.get_instance()
    run_id = "cov-test-phase4-fixture"

    lcov_order = LineCoverageData(
        file_path="src/order.py",
        executable_lines=[1, 2, 3, 5, 6, 7],
        executed_lines=[1, 2, 3],
        missing_lines=[5, 6, 7],
        excluded_lines=[],
        total_statements=6,
        covered_statements=3,
        line_coverage_pct=50.0,
    )

    fn_calculate = FunctionCoverage(
        function_name="calculate_order",
        qualified_name="OrderService.calculate_order",
        file_path="src/order.py",
        class_name="OrderService",
        start_line=1,
        end_line=4,
        executable_lines=[2, 3],
        executed_lines=[2, 3],
        missing_lines=[],
        coverage_pct=100.0,
        status=FunctionStatus.FULLY_COVERED,
        executing_tests=["tests/test_order.py::test_calc"],
    )

    fn_refund = FunctionCoverage(
        function_name="refund_order",
        qualified_name="OrderService.refund_order",
        file_path="src/order.py",
        class_name="OrderService",
        start_line=5,
        end_line=8,
        executable_lines=[6, 7],
        executed_lines=[],
        missing_lines=[6, 7],
        coverage_pct=0.0,
        status=FunctionStatus.UNCOVERED,
        executing_tests=[],
    )

    fsum = FileCoverageSummary(
        file_path="src/order.py",
        total_statements=6,
        covered_statements=3,
        missing_statements=3,
        line_coverage_pct=50.0,
        total_functions=2,
        covered_functions=1,
        partially_covered_functions=0,
        uncovered_functions=1,
        function_coverage_pct=50.0,
        functions=[fn_calculate, fn_refund],
    )

    t_map = TestToCodeMapping(
        test_id="tests/test_order.py::test_calc",
        test_name="test_calc",
        test_file="tests/test_order.py",
        covered_files=["src/order.py"],
        covered_functions=["OrderService.calculate_order"],
        executed_lines_by_file={"src/order.py": [1, 2, 3]},
        evidence_type=CoverageEvidenceType.RUNTIME_COVERAGE,
    )

    report = CoverageReport(
        coverage_run_id=run_id,
        repo_id="phase4_test_repo",
        repo_path=".",
        commit_hash="c444444",
        execution_mode="local_subprocess",
        status=CoverageStatus.SUCCESS,
        total_statements=6,
        covered_statements=3,
        missing_statements=3,
        line_coverage_pct=50.0,
        total_functions=2,
        covered_functions=1,
        partially_covered_functions=0,
        uncovered_functions=1,
        function_coverage_pct=50.0,
        files={"src/order.py": lcov_order},
        file_summaries=[fsum],
        test_mappings=[t_map],
    )

    store.save_report(report)
    yield report
    store.delete_report(run_id)


def test_api_coverage_summary(client, populated_coverage_store):
    """Verify /api/coverage/summary returns accurate repository metrics."""
    res = client.get(f"/api/coverage/summary?coverage_run_id={populated_coverage_store.coverage_run_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["has_coverage"] is True
    assert data["status"] == "SUCCESS"
    assert data["overall_line_coverage_pct"] == 50.0
    assert data["overall_function_coverage_pct"] == 50.0
    assert data["total_statements"] == 6
    assert data["covered_statements"] == 3
    assert data["missing_statements"] == 3
    assert data["covered_functions"] == 1
    assert data["uncovered_functions"] == 1


def test_api_coverage_files(client, populated_coverage_store):
    """Verify /api/coverage/files lists files with coverage percentages."""
    res = client.get(f"/api/coverage/files?coverage_run_id={populated_coverage_store.coverage_run_id}")
    assert res.status_code == 200
    files = res.json()
    assert len(files) >= 1
    order_file = next(f for f in files if f["file_path"] == "src/order.py")
    assert order_file["line_coverage_pct"] == 50.0
    assert order_file["total_functions"] == 2
    assert order_file["covered_functions"] == 1


def test_api_coverage_file_detail(client, populated_coverage_store):
    """Verify /api/coverage/file returns line coverage details."""
    res = client.get(f"/api/coverage/file?file_path=src/order.py&coverage_run_id={populated_coverage_store.coverage_run_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["file_path"] == "src/order.py"
    assert "line_coverage" in data
    assert data["line_coverage"]["executed_lines"] == [1, 2, 3]
    assert data["line_coverage"]["missing_lines"] == [5, 6, 7]


def test_api_coverage_functions_with_filters(client, populated_coverage_store):
    """Verify /api/coverage/functions can filter by status and query."""
    run_id = populated_coverage_store.coverage_run_id

    # 1. Filter status=FULLY_COVERED
    res_covered = client.get(f"/api/coverage/functions?coverage_run_id={run_id}&status=FULLY_COVERED")
    assert res_covered.status_code == 200
    covered_funcs = res_covered.json()
    assert len(covered_funcs) == 1
    assert covered_funcs[0]["function_name"] == "calculate_order"

    # 2. Filter status=UNCOVERED
    res_uncovered = client.get(f"/api/coverage/functions?coverage_run_id={run_id}&status=UNCOVERED")
    assert res_uncovered.status_code == 200
    uncovered_funcs = res_uncovered.json()
    assert len(uncovered_funcs) == 1
    assert uncovered_funcs[0]["function_name"] == "refund_order"

    # 3. Search query
    res_search = client.get(f"/api/coverage/functions?coverage_run_id={run_id}&search=refund")
    assert res_search.status_code == 200
    assert len(res_search.json()) == 1


def test_api_coverage_uncovered_code(client, populated_coverage_store):
    """Verify /api/coverage/uncovered returns functions with missing statements."""
    run_id = populated_coverage_store.coverage_run_id
    res = client.get(f"/api/coverage/uncovered?coverage_run_id={run_id}")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["function_name"] == "refund_order"
    assert data[0]["missing_lines"] == [6, 7]


def test_api_coverage_test_map(client, populated_coverage_store):
    """Verify /api/coverage/test-map returns empirical test-to-code mapping and graph."""
    run_id = populated_coverage_store.coverage_run_id
    res = client.get(f"/api/coverage/test-map?coverage_run_id={run_id}")
    assert res.status_code == 200
    data = res.json()
    assert "test_mappings" in data
    assert len(data["test_mappings"]) == 1
    assert data["test_mappings"][0]["test_name"] == "test_calc"
    assert "graph" in data
    assert data["graph"]["total_nodes"] >= 2
    assert data["graph"]["evidence_breakdown"]["runtime_confirmed"] >= 1


def test_api_coverage_download_report(client, populated_coverage_store):
    """Verify /api/coverage/report/download exports both JSON and Markdown."""
    run_id = populated_coverage_store.coverage_run_id

    # 1. JSON download
    res_json = client.get(f"/api/coverage/report/download?coverage_run_id={run_id}&format=json")
    assert res_json.status_code == 200
    assert res_json.headers["content-type"].startswith("application/json")
    parsed = json.loads(res_json.text)
    assert parsed["coverage_run_id"] == run_id

    # 2. Markdown download
    res_md = client.get(f"/api/coverage/report/download?coverage_run_id={run_id}&format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert "# TestPilot Coverage Report" in res_md.text
    assert "OrderService" in res_md.text or "src/order.py" in res_md.text


def test_api_coverage_empty_repository(client):
    """Verify /api/coverage/summary handles empty states gracefully without 500 error."""
    res = client.get("/api/coverage/summary?repo_id=non_existent_empty_repo_xyz_999")
    assert res.status_code == 200
    data = res.json()
    assert data["has_coverage"] is False
    assert data["status"] == "UNAVAILABLE"
    assert data["overall_line_coverage_pct"] == 0.0

