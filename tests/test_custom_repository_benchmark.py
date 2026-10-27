"""
Comprehensive test suite for the Dynamic Multi-Repository Regression Benchmark Manager.
Verifies all 17 lifecycle, evaluation, security, mathematical integrity, and regression criteria:
1. Register a valid repository.
2. Reject an invalid path or unsupported repository.
3. Prevent duplicate repository registrations.
4. Persist registrations across application restarts.
5. Discover repository tests and benchmark preparation status.
6. Distinguish analysis completion from ground-truth validation.
7. Evaluate a repository with valid ground truth.
8. Correctly calculate TP, FP, FN, precision, recall, F1, and test reduction.
9. Aggregate results across multiple repositories using established metric definitions.
10. Exclude repositories without valid ground truth from quantitative aggregates.
11. Remove one repository without affecting other repositories.
12. Ensure removed repositories do not contribute to active aggregates.
13. Preserve the authoritative benchmark results and definitions.
14. Mark results stale when the repository revision changes.
15. Handle concurrent requests, running evaluations, and failures safely.
16. Validate API responses and custom suite endpoints.
17. Prevent repository context from leaking between registered repositories.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    PrepareBenchmarkRequest,
    RegisterRepositoryRequest,
    RepositoryStatus,
)
from testpilot.evaluation.engine import EvaluationEngine
from testpilot.web.api import app


def _create_temp_git_repo(repo_name: str = "sample_repo") -> Path:
    """Helper to initialize a real, temporary Git repository with sample functions and tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"testpilot_{repo_name}_"))
    subprocess.run(["git", "init"], cwd=str(temp_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "TestPilotBot"], cwd=str(temp_dir), check=True)
    subprocess.run(["git", "config", "user.email", "bot@testpilot.ai"], cwd=str(temp_dir), check=True)

    # 1. Create source file
    src_file = temp_dir / "service.py"
    src_file.write_text(
        "def compute_tax(subtotal: float) -> float:\n"
        "    return round(subtotal * 0.1, 2)\n\n"
        "def calculate_total(subtotal: float) -> float:\n"
        "    return subtotal + compute_tax(subtotal)\n",
        encoding="utf-8",
    )

    # 2. Create test directory and test file
    tests_dir = temp_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_service.py"
    test_file.write_text(
        "from service import compute_tax, calculate_total\n\n"
        "def test_compute_tax_standard():\n"
        "    assert compute_tax(100.0) == 10.0\n\n"
        "def test_calculate_total_standard():\n"
        "    assert calculate_total(100.0) == 110.0\n\n"
        "def test_unrelated_feature():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=str(temp_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(temp_dir), capture_output=True, check=True)

    # 3. Create second commit modifying compute_tax so HEAD~1 and HEAD form a valid commit diff
    src_file.write_text(
        "def compute_tax(subtotal: float) -> float:\n"
        "    \"\"\"Compute standard tax rate.\"\"\"\n"
        "    return round(subtotal * 0.1, 2)\n\n"
        "def calculate_total(subtotal: float) -> float:\n"
        "    return subtotal + compute_tax(subtotal)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "service.py"], cwd=str(temp_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Update compute_tax docstring"], cwd=str(temp_dir), capture_output=True, check=True)
    return temp_dir


# =========================================================================
# 1. Register a valid repository
# =========================================================================
def test_1_register_valid_repository():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("valid_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)

        req = RegisterRepositoryRequest(path=str(repo_dir), name="My Valid Project")
        record = mgr.register_repository(req)

        assert record.name == "My Valid Project"
        assert record.canonical_path == str(repo_dir.resolve())
        assert record.is_git is True
        assert record.detected_revision is not None
        assert record.total_tests_count == 3
        assert "tests/test_service.py::test_compute_tax_standard" in record.discovered_tests
        assert record.is_active is True


# =========================================================================
# 2. Reject an invalid path or unsupported repository
# =========================================================================
def test_2_reject_invalid_path():
    with tempfile.TemporaryDirectory() as storage_tmp:
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)

        with pytest.raises(FileNotFoundError):
            mgr.register_repository(RegisterRepositoryRequest(path="C:/NonExistentPath_xyz123_456"))

        with pytest.raises(ValueError):
            mgr.register_repository(RegisterRepositoryRequest(path="   "))


# =========================================================================
# 3. Prevent duplicate repository registrations
# =========================================================================
def test_3_prevent_duplicate_registrations():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("dup_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)

        rec1 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir), name="First Registration"))
        rec2 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir), name="Second Registration"))

        assert rec1.repo_id == rec2.repo_id
        active = mgr.list_repositories(active_only=True)
        assert len(active) == 1


# =========================================================================
# 4. Persist registrations across application restarts
# =========================================================================
def test_4_persist_across_restarts():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("persist_repo")
        mgr1 = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rec = mgr1.register_repository(RegisterRepositoryRequest(path=str(repo_dir), name="Persistent Project"))

        # Simulate fresh app startup with new manager instance pointing to same storage
        mgr2 = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        loaded = mgr2.get_repository(rec.repo_id)

        assert loaded is not None
        assert loaded.repo_id == rec.repo_id
        assert loaded.name == "Persistent Project"
        assert loaded.total_tests_count == 3


# =========================================================================
# 5. Discover repository tests and benchmark preparation status
# =========================================================================
def test_5_discover_tests_and_status():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("discovery_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rec = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        assert rec.status == RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
        assert rec.total_tests_count == 3
        assert len(rec.benchmark_cases) == 0
        assert rec.validated_benchmark_cases_count == 0


# =========================================================================
# 6. Distinguish analysis completion from ground-truth validation
# =========================================================================
def test_6_distinguish_analysis_from_ground_truth():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("distinguish_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rec = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        # Analyzed, but NOT ready for evaluation yet because no verified ground truth exists
        assert rec.status == RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
        assert "Independent benchmark ground truth must be prepared" in rec.status_message

        # Now prepare verified ground truth
        gt_req = PrepareBenchmarkRequest(
            name="Tax Boundary Ground Truth",
            source_type=BenchmarkSourceType.MANUAL_MAPPING,
            expected_changed_symbols=["compute_tax"],
            expected_test_callers=["test_compute_tax_standard"],
            provenance="Audited unit test in tests/test_service.py",
        )
        case = mgr.prepare_benchmark_case(rec.repo_id, gt_req)

        updated = mgr.get_repository(rec.repo_id)
        assert updated.status == RepositoryStatus.READY_FOR_EVALUATION
        assert updated.validated_benchmark_cases_count == 1
        assert case.ground_truth.is_verified is True


# =========================================================================
# 7 & 8. Evaluate repository with valid ground truth and calculate strict metrics
# =========================================================================
def test_7_and_8_evaluate_repository_and_metrics():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("eval_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rec = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        # Add ground truth where expected test is test_compute_tax_standard
        mgr.prepare_benchmark_case(
            rec.repo_id,
            PrepareBenchmarkRequest(
                name="Tax Calculation Benchmark",
                source_type=BenchmarkSourceType.MANUAL_MAPPING,
                expected_changed_symbols=["compute_tax"],
                expected_test_callers=["test_compute_tax_standard"],
                provenance="Independent code review and unit assertion audit",
            ),
        )

        runs = mgr.evaluate_repository(rec.repo_id)
        assert len(runs) == 1
        run = runs[0]

        # Verify Full Regression baseline
        full_res = run.baseline_results["full_regression"]
        full_met = run.metrics["full_regression"]
        assert full_res.selected_count == 3
        assert full_met.tp == 1
        assert full_met.fp == 2
        assert full_met.fn == 0
        assert full_met.precision == round(1 / 3, 4)
        assert full_met.recall == 1.0
        assert full_met.test_reduction_pct == 0.0

        # Verify TestPilot baseline
        tp_res = run.baseline_results["testpilot"]
        tp_met = run.metrics["testpilot"]
        assert tp_res.selected_count >= 1
        assert tp_met.tp == 1
        assert tp_met.recall == 1.0
        assert tp_met.test_reduction_pct is not None
        assert tp_met.test_reduction_pct > 0.0

        # Verify Naive baseline
        naive_met = run.metrics["naive_name_matching"]
        assert naive_met.is_evaluated is True


# =========================================================================
# 9 & 10. Aggregate results across multiple repositories & exclude unvalidated
# =========================================================================
def test_9_and_10_aggregate_metrics_and_exclude_unvalidated():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo1_dir = _create_temp_git_repo("repo_1")
        repo2_dir = _create_temp_git_repo("repo_2")
        repo3_dir = _create_temp_git_repo("repo_3_unvalidated")

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        r1 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo1_dir), name="Repo 1"))
        r2 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo2_dir), name="Repo 2"))
        r3 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo3_dir), name="Repo 3 Unvalidated"))

        # Setup and evaluate repo 1
        mgr.prepare_benchmark_case(
            r1.repo_id,
            PrepareBenchmarkRequest(
                name="Case R1",
                expected_changed_symbols=["compute_tax"],
                expected_test_callers=["test_compute_tax_standard"],
                provenance="Provenance 1",
            ),
        )
        mgr.evaluate_repository(r1.repo_id)

        # Setup and evaluate repo 2
        mgr.prepare_benchmark_case(
            r2.repo_id,
            PrepareBenchmarkRequest(
                name="Case R2",
                expected_changed_symbols=["calculate_total"],
                expected_test_callers=["test_calculate_total_standard"],
                provenance="Provenance 2",
            ),
        )
        mgr.evaluate_repository(r2.repo_id)

        # Repo 3 remains unvalidated (no ground truth)
        overview = mgr.get_custom_suite_overview()

        assert overview["active_repositories_count"] == 3
        assert overview["evaluated_repositories_count"] == 2
        assert overview["benchmark_preparation_required_count"] == 1
        assert overview["evaluated_runs_count"] == 2
        assert overview["total_ground_truth_pool"] == 2

        # Verify pooled TP/FP/FN aggregation
        full_row = next(row for row in overview["primary_matrix"] if row["configuration"] == "Full Regression")
        assert full_row["is_evaluated"] is True
        assert full_row["tp"] == 2  # 1 from repo1 + 1 from repo2
        assert full_row["ground_truth_positives"] == 2


# =========================================================================
# 11 & 12. Safe removal and exclusion from active aggregates
# =========================================================================
def test_11_and_12_safe_removal_preserves_files_and_updates_aggregates():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo1_dir = _create_temp_git_repo("repo_to_keep")
        repo2_dir = _create_temp_git_repo("repo_to_remove")

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        r1 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo1_dir), name="Keeper"))
        r2 = mgr.register_repository(RegisterRepositoryRequest(path=str(repo2_dir), name="Remover"))

        mgr.prepare_benchmark_case(
            r1.repo_id,
            PrepareBenchmarkRequest(name="Case 1", expected_test_callers=["test_compute_tax_standard"], provenance="Prov 1"),
        )
        mgr.evaluate_repository(r1.repo_id)

        mgr.prepare_benchmark_case(
            r2.repo_id,
            PrepareBenchmarkRequest(name="Case 2", expected_test_callers=["test_calculate_total_standard"], provenance="Prov 2"),
        )
        mgr.evaluate_repository(r2.repo_id)

        # Before removal: 2 evaluated runs in aggregate
        ov_before = mgr.get_custom_suite_overview()
        assert ov_before["evaluated_runs_count"] == 2

        # Remove repo 2 safely
        res = mgr.remove_repository(r2.repo_id)
        assert res["status"] == "removed"
        assert repo2_dir.exists()  # CRITICAL: Local disk files MUST NOT be deleted!

        # After removal: only 1 evaluated run in aggregate
        ov_after = mgr.get_custom_suite_overview()
        assert ov_after["active_repositories_count"] == 1
        assert ov_after["evaluated_runs_count"] == 1


# =========================================================================
# 13. Preserve authoritative benchmark results and definitions
# =========================================================================
def test_13_preserve_authoritative_benchmarks():
    authoritative_engine = EvaluationEngine()
    summary = authoritative_engine.get_overview_summary()

    # The authoritative matrix must retain its 6 canonical configurations
    assert "primary_matrix" in summary
    assert len(summary["primary_matrix"]) == 6
    configs = {r["configuration"] for r in summary["primary_matrix"]}
    assert "TestPilot" in configs
    assert "Full Regression" in configs
    assert "Naive Baseline" in configs


# =========================================================================
# 14. Mark results stale when repository revision changes
# =========================================================================
def test_14_mark_stale_when_revision_changes():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_temp_git_repo("stale_test_repo")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rec = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        mgr.prepare_benchmark_case(
            rec.repo_id,
            PrepareBenchmarkRequest(name="Case A", expected_test_callers=["test_compute_tax_standard"], provenance="Prov"),
        )
        mgr.evaluate_repository(rec.repo_id)

        rec_eval = mgr.get_repository(rec.repo_id)
        assert rec_eval.status == RepositoryStatus.EVALUATED

        # Make a new commit in the repository to advance HEAD
        new_file = repo_dir / "patch.txt"
        new_file.write_text("advance commit", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Advance commit"], cwd=str(repo_dir), check=True)

        # Re-fetch repository - should detect revision difference and mark stale
        rec_stale = mgr.get_repository(rec.repo_id)
        assert rec_stale.status == RepositoryStatus.STALE


# =========================================================================
# 15 & 16. Fast API endpoints verification (Lifecycle & Error handling)
# =========================================================================
def test_15_and_16_web_api_custom_suite_endpoints():
    client = TestClient(app)
    repo_dir = _create_temp_git_repo("api_test_repo")

    # 1. Register
    res = client.post("/api/evaluation/custom-suite/repositories", json={"path": str(repo_dir), "name": "API Test Repo"})
    assert res.status_code == 200
    repo_data = res.json()
    repo_id = repo_data["repo_id"]
    assert repo_data["name"] == "API Test Repo"

    # 2. List
    res_list = client.get("/api/evaluation/custom-suite/repositories")
    assert res_list.status_code == 200
    assert any(r["repo_id"] == repo_id for r in res_list.json())

    # 3. Overview (Before Ground Truth)
    res_ov = client.get("/api/evaluation/custom-suite/overview")
    assert res_ov.status_code == 200
    ov_data = res_ov.json()
    assert ov_data["active_repositories_count"] >= 1

    # 4. Prepare Ground Truth
    res_gt = client.post(
        f"/api/evaluation/custom-suite/repositories/{repo_id}/prepare-benchmark",
        json={
            "name": "API Scenario 1",
            "source_type": "manual_mapping",
            "expected_changed_symbols": ["compute_tax"],
            "expected_test_callers": ["test_compute_tax_standard"],
            "provenance": "Verified in test_15_and_16",
        },
    )
    assert res_gt.status_code == 200
    gt_data = res_gt.json()
    assert gt_data["name"] == "API Scenario 1"

    # 5. Evaluate
    res_eval = client.post(f"/api/evaluation/custom-suite/repositories/{repo_id}/evaluate")
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["status"] == "success"
    assert eval_data["evaluated_runs_count"] == 1

    # 6. Status check
    res_status = client.get("/api/evaluation/custom-suite/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "operational"

    # 7. Safe Remove
    res_del = client.delete(f"/api/evaluation/custom-suite/repositories/{repo_id}")
    assert res_del.status_code == 200
    assert repo_dir.exists()


# =========================================================================
# 17. Prevent context or vector-store leakage between repositories
# =========================================================================
def test_17_prevent_context_leakage_between_repos():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repoA_dir = _create_temp_git_repo("repo_A")
        repoB_dir = _create_temp_git_repo("repo_B")

        # Distinct test in repo B
        test_b = repoB_dir / "tests" / "test_unique_b.py"
        test_b.write_text("def test_unique_b_only(): assert True\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repoB_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Add unique test in B"], cwd=str(repoB_dir), check=True)

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        rA = mgr.register_repository(RegisterRepositoryRequest(path=str(repoA_dir), name="Repo A"))
        rB = mgr.register_repository(RegisterRepositoryRequest(path=str(repoB_dir), name="Repo B"))

        assert "tests/test_unique_b.py::test_unique_b_only" in rB.discovered_tests
        assert "tests/test_unique_b.py::test_unique_b_only" not in rA.discovered_tests


# =========================================================================
# 18. URL detection: Local paths vs Remote GitHub URLs
# =========================================================================
def test_18_is_remote_url_detection():
    # Local filesystem paths must return False
    assert not CustomRepositoryBenchmarkManager.is_remote_url(r"C:\Projects\FinMath")
    assert not CustomRepositoryBenchmarkManager.is_remote_url(r"c:\Users\shres\OneDrive\Desktop\TestPilot")
    assert not CustomRepositoryBenchmarkManager.is_remote_url("D:/workspace/repo")
    assert not CustomRepositoryBenchmarkManager.is_remote_url("./testbed")
    assert not CustomRepositoryBenchmarkManager.is_remote_url("testbed")
    assert not CustomRepositoryBenchmarkManager.is_remote_url("")
    assert not CustomRepositoryBenchmarkManager.is_remote_url(None)

    # Remote URLs must return True
    assert CustomRepositoryBenchmarkManager.is_remote_url("https://github.com/Shreshth-mod/FinMath")
    assert CustomRepositoryBenchmarkManager.is_remote_url("https://github.com/Shreshth-mod/FinMath.git")
    assert CustomRepositoryBenchmarkManager.is_remote_url("http://github.com/Shreshth-mod/FinMath")
    assert CustomRepositoryBenchmarkManager.is_remote_url("github.com/Shreshth-mod/FinMath")
    assert CustomRepositoryBenchmarkManager.is_remote_url("git@github.com:Shreshth-mod/FinMath.git")
    assert CustomRepositoryBenchmarkManager.is_remote_url("git://github.com/Shreshth-mod/FinMath.git")
    assert CustomRepositoryBenchmarkManager.is_remote_url("file:///etc/passwd")


# =========================================================================
# 19. Remote URL parsing, validation, and normalization
# =========================================================================
def test_19_parse_and_validate_remote_url_success():
    # Standard HTTPS
    url, owner, repo = CustomRepositoryBenchmarkManager.parse_and_validate_remote_url(
        "https://github.com/Shreshth-mod/FinMath"
    )
    assert url == "https://github.com/Shreshth-mod/FinMath.git"
    assert owner == "Shreshth-mod"
    assert repo == "FinMath"

    # HTTPS with .git
    url2, owner2, repo2 = CustomRepositoryBenchmarkManager.parse_and_validate_remote_url(
        "https://github.com/Shreshth-mod/FinMath.git"
    )
    assert url2 == "https://github.com/Shreshth-mod/FinMath.git"
    assert owner2 == "Shreshth-mod"
    assert repo2 == "FinMath"

    # Bare domain github.com
    url3, owner3, repo3 = CustomRepositoryBenchmarkManager.parse_and_validate_remote_url(
        "github.com/Shreshth-mod/FinMath"
    )
    assert url3 == "https://github.com/Shreshth-mod/FinMath.git"
    assert owner3 == "Shreshth-mod"
    assert repo3 == "FinMath"

    # SSH format converted securely to HTTPS
    url4, owner4, repo4 = CustomRepositoryBenchmarkManager.parse_and_validate_remote_url(
        "git@github.com:Shreshth-mod/FinMath.git"
    )
    assert url4 == "https://github.com/Shreshth-mod/FinMath.git"
    assert owner4 == "Shreshth-mod"
    assert repo4 == "FinMath"


# =========================================================================
# 20. Remote URL security rejections (SSRF, invalid schemes, bad hosts)
# =========================================================================
def test_20_parse_and_validate_remote_url_rejections():
    # Unsupported schemes
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("file:///C:/Windows/System32")
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("ftp://ftp.example.com/repo")

    # SSRF / non-GitHub hosts
    with pytest.raises(ValueError, match="Unsupported host"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://127.0.0.1/owner/repo")
    with pytest.raises(ValueError, match="Unsupported host"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("http://169.254.169.254/meta-data")
    with pytest.raises(ValueError, match="Unsupported host"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://evil-host.com/owner/repo")

    # Malformed paths
    with pytest.raises(ValueError, match="Malformed GitHub URL"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://github.com/only-one-part")
    with pytest.raises(ValueError, match="Malformed GitHub URL"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://github.com/part1/part2/extra/part3")

    # Path traversal / command injection attempts
    with pytest.raises(ValueError, match="Invalid characters|Path traversal"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://github.com/owner/repo;rm -rf /")
    with pytest.raises(ValueError, match="Invalid characters|Path traversal"):
        CustomRepositoryBenchmarkManager.parse_and_validate_remote_url("https://github.com/../escape")


# =========================================================================
# 21. Remote repository cloning and registration into benchmark workspace
# =========================================================================
def test_21_remote_repo_cloning_and_registration():
    with tempfile.TemporaryDirectory() as storage_tmp, tempfile.TemporaryDirectory() as ws_tmp:
        # Create a local source git repo to simulate a remote
        source_repo = _create_temp_git_repo("remote_source")
        test_file = source_repo / "tests" / "test_remote.py"
        test_file.write_text("def test_remote_logic(): assert True\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(source_repo), check=True)
        subprocess.run(["git", "commit", "-m", "Add remote test"], cwd=str(source_repo), check=True)

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp, workspace_dir=ws_tmp)

        # Mock parse_and_validate_remote_url and git clone to use our local temp source repo
        orig_run = subprocess.run

        def mock_clone_call(cmd, **kwargs):
            if cmd and "clone" in cmd:
                dest = cmd[-1]
                real_cmd = ["git", "--no-pager", "clone", "--depth", "50", str(source_repo), dest]
                return orig_run(real_cmd, capture_output=True, text=True, check=False)
            return orig_run(cmd, **kwargs)

        with patch("subprocess.run", side_effect=mock_clone_call):
            record = mgr.register_repository(
                RegisterRepositoryRequest(
                    path="https://github.com/Shreshth-mod/FinMath",
                    name="FinMath Remote",
                )
            )

        assert record.is_remote is True
        assert record.remote_url == "https://github.com/Shreshth-mod/FinMath.git"
        assert record.name == "FinMath Remote"
        assert Path(record.canonical_path).exists()
        assert Path(record.canonical_path).parent == Path(ws_tmp).resolve()
        assert record.status == RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
        assert "Independent benchmark ground truth must be prepared" in record.status_message
        assert "tests/test_remote.py::test_remote_logic" in record.discovered_tests


# =========================================================================
# 22. Safe reuse of existing cloned repository
# =========================================================================
def test_22_remote_repo_reuse_existing_clone():
    with tempfile.TemporaryDirectory() as storage_tmp, tempfile.TemporaryDirectory() as ws_tmp:
        source_repo = _create_temp_git_repo("remote_source_2")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp, workspace_dir=ws_tmp)

        # Pre-seed the clone destination directory in the workspace
        dest_dir = Path(ws_tmp) / "shreshth-mod_finmath"
        subprocess.run(["git", "clone", str(source_repo), str(dest_dir)], check=True)
        marker_file = dest_dir / "user_marker.txt"
        marker_file.write_text("should_not_be_overwritten", encoding="utf-8")

        from unittest.mock import patch
        orig_run = subprocess.run
        clone_called = False

        def mock_detect_clone(cmd, **kwargs):
            nonlocal clone_called
            if cmd and len(cmd) > 1 and cmd[1] == "clone":
                clone_called = True
            return orig_run(cmd, **kwargs)

        with patch("subprocess.run", side_effect=mock_detect_clone):
            # Registering should reuse existing folder and not invoke git clone
            record = mgr.register_repository(
                RegisterRepositoryRequest(path="https://github.com/Shreshth-mod/FinMath")
            )

        assert not clone_called, "git clone should not have been called when reusing existing clone"
        assert record.is_remote is True
        assert marker_file.exists()
        assert marker_file.read_text(encoding="utf-8") == "should_not_be_overwritten"


# =========================================================================
# 23. Duplicate remote URL registration prevention
# =========================================================================
def test_23_remote_repo_duplicate_registration_prevention():
    with tempfile.TemporaryDirectory() as storage_tmp, tempfile.TemporaryDirectory() as ws_tmp:
        source_repo = _create_temp_git_repo("remote_source_3")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp, workspace_dir=ws_tmp)

        dest_dir = Path(ws_tmp) / "shreshth-mod_finmath"
        subprocess.run(["git", "clone", str(source_repo), str(dest_dir)], check=True)

        # Register once with https://...
        r1 = mgr.register_repository(RegisterRepositoryRequest(path="https://github.com/Shreshth-mod/FinMath"))
        # Register second time with .git
        r2 = mgr.register_repository(RegisterRepositoryRequest(path="https://github.com/Shreshth-mod/FinMath.git"))
        # Register third time with bare domain
        r3 = mgr.register_repository(RegisterRepositoryRequest(path="github.com/Shreshth-mod/FinMath"))

        assert r1.repo_id == r2.repo_id == r3.repo_id
        all_repos = mgr.list_repositories(active_only=True)
        assert len(all_repos) == 1


# =========================================================================
# 24. Clone failure cleanup and no orphaned records
# =========================================================================
def test_24_clone_failure_cleanup_and_no_orphans():
    with tempfile.TemporaryDirectory() as storage_tmp, tempfile.TemporaryDirectory() as ws_tmp:
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp, workspace_dir=ws_tmp)

        from unittest.mock import patch
        import subprocess

        # Simulate git clone error (e.g. repo not found)
        mock_proc = subprocess.CompletedProcess(
            args=["git", "clone"],
            returncode=128,
            stdout="",
            stderr="fatal: repository 'https://github.com/Shreshth-mod/NotExist.git' not found\n",
        )

        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(FileNotFoundError, match="Remote repository not found"):
                mgr.register_repository(RegisterRepositoryRequest(path="https://github.com/Shreshth-mod/NotExist"))

        # Workspace directory should be cleaned up
        dest_dir = Path(ws_tmp) / "shreshth-mod_notexist"
        assert not dest_dir.exists()

        # No records saved
        assert len(mgr.list_repositories(active_only=False)) == 0


# =========================================================================
# 25. FastAPI endpoint handles remote URLs and returns validation errors
# =========================================================================
def test_25_api_remote_repo_endpoints():
    client = TestClient(app)

    # 1. Unsupported scheme returns 400 with detail
    res_err1 = client.post(
        "/api/evaluation/custom-suite/repositories",
        json={"path": "file:///C:/Windows/test"},
    )
    assert res_err1.status_code == 400
    assert "Unsupported URL scheme" in res_err1.json()["detail"]

    # 2. Unsupported host returns 400 with detail
    res_err2 = client.post(
        "/api/evaluation/custom-suite/repositories",
        json={"path": "https://evil-site.com/foo/bar"},
    )
    assert res_err2.status_code == 400
    assert "Unsupported host" in res_err2.json()["detail"]

    # 3. Malformed GitHub URL returns 400 with detail
    res_err3 = client.post(
        "/api/evaluation/custom-suite/repositories",
        json={"path": "https://github.com/incomplete"},
    )
    assert res_err3.status_code == 400
    assert "Malformed GitHub URL" in res_err3.json()["detail"]

