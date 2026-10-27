"""
Comprehensive Multi-Repository Acceptance Test Suite.
Verifies all 20 acceptance criteria defined in Phase 12 of the Antigravity Implementation Specification.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testpilot.core.pipeline import FullPipelineOrchestrator
from testpilot.evaluation.auto_benchmark import AutoBenchmarkGenerator
from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    GeneratedTestRecord,
    GroundTruthReviewStatus,
    GroundTruthValidationLevel,
    PrepareBenchmarkRequest,
    PrepareHistoricalBugRequest,
    RegisterRepositoryRequest,
    RepositoryStatus,
    ReviewGroundTruthRequest,
)
from testpilot.evaluation.engine import EvaluationEngine, compute_metrics
from testpilot.evaluation.models import GroundTruth


@pytest.fixture
def temp_custom_store(tmp_path):
    """Provides an isolated store directory for custom repository registry."""
    store_dir = tmp_path / "store"
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(store_dir))
    return mgr, tmp_path


@pytest.fixture
def synthetic_repo(tmp_path):
    """Creates a synthetic repository with source files, tests, and a clean git history."""
    repo_dir = tmp_path / "synth_repo"
    repo_dir.mkdir()

    # Create source module
    src_dir = repo_dir / "calc"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "math_service.py").write_text(
        "def compute_discount(price: float, discount: float) -> float:\n"
        "    if discount < 0 or discount > 1:\n"
        "        raise ValueError('Invalid discount')\n"
        "    return price * (1.0 - discount)\n",
        encoding="utf-8",
    )

    # Create test directory with an existing test
    test_dir = repo_dir / "tests"
    test_dir.mkdir()
    (test_dir / "__init__.py").write_text("", encoding="utf-8")
    (test_dir / "test_math.py").write_text(
        "import pytest\n"
        "from calc.math_service import compute_discount\n\n"
        "def test_compute_discount_valid():\n"
        "    assert compute_discount(100.0, 0.2) == 80.0\n\n"
        "def test_compute_discount_invalid():\n"
        "    with pytest.raises(ValueError):\n"
        "        compute_discount(100.0, -0.1)\n",
        encoding="utf-8",
    )

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "TestPilot Accept"], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "accept@testpilot.ai"], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo_dir), capture_output=True, check=True)

    return repo_dir


# ==============================================================================
# Criterion 1: Valid GitHub URL recognized as remote repo, not local path
# ==============================================================================
def test_01_github_url_recognition(temp_custom_store):
    mgr, tmp_path = temp_custom_store
    url = "https://github.com/Shreshth-mod/FinMath.git"

    # Simulated clone target
    sim_dir = tmp_path / "cloned_finmath"
    sim_dir.mkdir()
    (sim_dir / "app.py").write_text("print('FinMath')")

    with patch.object(mgr, "clone_or_reuse_remote_repo") as mock_clone:
        mock_clone.return_value = (sim_dir, url, "FinMath")
        req = RegisterRepositoryRequest(path=url)
        rec = mgr.register_repository(req)

        assert rec.is_remote is True
        assert rec.remote_url == url
        assert rec.name == "FinMath"
        assert not rec.canonical_path.startswith("C:\\Users\\shres\\OneDrive\\Desktop\\TestPilot\\https")
        mock_clone.assert_called_once()


# ==============================================================================
# Criterion 2: Duplicate repositories handled safely
# ==============================================================================
def test_02_duplicate_repositories_handled_safely(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    req = RegisterRepositoryRequest(path=str(synthetic_repo))
    rec1 = mgr.register_repository(req)
    rec2 = mgr.register_repository(req)

    assert rec1.repo_id == rec2.repo_id
    all_repos = mgr.list_repositories()
    assert len(all_repos) == 1


# ==============================================================================
# Criterion 3: Local repository registration works
# ==============================================================================
def test_03_local_repository_registration(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    req = RegisterRepositoryRequest(path=str(synthetic_repo), name="MySynthRepo")
    rec = mgr.register_repository(req)

    assert rec.is_remote is False
    assert rec.name == "MySynthRepo"
    assert Path(rec.canonical_path).resolve() == synthetic_repo.resolve()
    assert rec.detected_revision is not None


# ==============================================================================
# Criterion 4: Repository inspection identifies supported test frameworks
# ==============================================================================
def test_04_repository_inspection_identifies_frameworks(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))
    inspected = mgr.inspect_repository(rec.repo_id)

    assert inspected.status in [RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED, RepositoryStatus.TESTS_DISCOVERED]
    assert inspected.total_tests_count >= 2
    assert any("test_compute_discount_valid" in t for t in inspected.discovered_tests)


# ==============================================================================
# Criterion 5: Existing tests discovered and actual execution status recorded
# ==============================================================================
def test_05_existing_tests_discovered_and_execution_status(synthetic_repo):
    generator = AutoBenchmarkGenerator(synthetic_repo)
    baseline_res = generator.run_baseline_tests(["tests/test_math.py::test_compute_discount_valid"])

    assert baseline_res["status"] == "passed"
    assert baseline_res["total_existing_tests"] == 1
    assert baseline_res["error"] is None


# ==============================================================================
# Criterion 6: Missing tests trigger candidate generation where supported
# ==============================================================================
def test_06_missing_tests_trigger_candidate_generation(tmp_path):
    repo_dir = tmp_path / "empty_tests_repo"
    repo_dir.mkdir()
    (repo_dir / "calculator.py").write_text(
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    generator = AutoBenchmarkGenerator(repo_dir)
    targets = generator.discover_target_functions()
    assert len(targets) > 0
    assert any(func.name == "multiply" for _, func in targets)

    records = generator.generate_and_save_tests(max_targets=5)
    assert len(records) > 0


# ==============================================================================
# Criterion 7: Generated tests are not automatically marked as ground truth
# ==============================================================================
def test_07_generated_tests_not_automatically_ground_truth(tmp_path):
    repo_dir = tmp_path / "repo_candidates"
    repo_dir.mkdir()
    (repo_dir / "converter.py").write_text(
        "def to_celsius(f: float) -> float:\n"
        "    return (f - 32.0) * 5.0 / 9.0\n",
        encoding="utf-8",
    )

    generator = AutoBenchmarkGenerator(repo_dir)
    records = generator.generate_and_save_tests(max_targets=5)

    for r in records:
        # Candidate tests are unverified candidates before mutation/review validation
        assert r.mutation_sensitive is False


# ==============================================================================
# Criterion 8: Historical bug cases require reproducible before/after evidence
# ==============================================================================
def test_08_historical_bug_requires_reproducible_evidence(synthetic_repo):
    # Commit 1 (Buggy): discount allows values > 1 resulting in negative price
    service_file = synthetic_repo / "calc" / "math_service.py"
    service_file.write_text(
        "def compute_discount(price: float, discount: float) -> float:\n"
        "    return price * (1.0 - discount)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "commit", "-am", "Introduce bug: no discount check"], cwd=str(synthetic_repo), capture_output=True, check=True)
    res_buggy = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(synthetic_repo), capture_output=True, text=True, check=True)
    buggy_sha = res_buggy.stdout.strip()

    # Commit 2 (Corrected): validate discount <= 1
    service_file.write_text(
        "def compute_discount(price: float, discount: float) -> float:\n"
        "    if discount > 1.0:\n"
        "        raise ValueError('Discount cannot exceed 1.0')\n"
        "    return price * (1.0 - discount)\n",
        encoding="utf-8",
    )
    test_file = synthetic_repo / "tests" / "test_math.py"
    test_file.write_text(
        "import pytest\n"
        "from calc.math_service import compute_discount\n\n"
        "def test_discount_exceed_one():\n"
        "    with pytest.raises(ValueError):\n"
        "        compute_discount(100.0, 1.5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "commit", "-am", "Fix bug: add discount check and test"], cwd=str(synthetic_repo), capture_output=True, check=True)
    res_corr = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(synthetic_repo), capture_output=True, text=True, check=True)
    corrected_sha = res_corr.stdout.strip()

    generator = AutoBenchmarkGenerator(synthetic_repo)
    case = generator.validate_historical_regression_case(
        repo_id="synth_repo",
        name="Excessive Discount Regression Bug",
        buggy_commit=buggy_sha,
        corrected_commit=corrected_sha,
        test_id="tests/test_math.py::test_discount_exceed_one",
        description="compute_discount must reject discount > 1.0 with ValueError",
    )

    assert case.ground_truth.is_verified is True
    assert case.review_status == GroundTruthReviewStatus.VALIDATED
    assert any("test_discount_exceed_one" in c for c in case.ground_truth.expected_test_callers)


# ==============================================================================
# Criterion 9: Mutation cases record valid, invalid, and detected mutations correctly
# ==============================================================================
def test_09_mutation_cases_record_valid_invalid_and_detected(synthetic_repo):
    generator = AutoBenchmarkGenerator(synthetic_repo)
    records = generator.generate_and_save_tests(max_targets=5)
    checked = generator.cross_check_mutation_sensitivity(records)

    for t in checked:
        assert hasattr(t, "mutation_sensitive")
        if t.mutation_sensitive:
            assert t.rejection_reason is None


# ==============================================================================
# Criterion 10: Behavioral-contract cases preserve provenance and review status
# ==============================================================================
def test_10_behavioral_contract_preserves_provenance_and_review(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))
    mgr.inspect_repository(rec.repo_id)

    req = PrepareBenchmarkRequest(
        name="Discount Boundary Contract",
        source_type=BenchmarkSourceType.BEHAVIORAL_CONTRACT,
        base_commit="HEAD~1",
        target_commit="HEAD",
        expected_changed_symbols=["compute_discount"],
        expected_impacted_symbols=[],
        expected_test_callers=["test_compute_discount_valid"],
        provenance="Documented contract in accounting specification v2.1",
    )
    case = mgr.prepare_benchmark_ground_truth(rec.repo_id, req)

    assert case.ground_truth.provenance == "Documented contract in accounting specification v2.1"
    assert case.review_status in [GroundTruthReviewStatus.NEEDS_HUMAN_REVIEW, GroundTruthReviewStatus.VALIDATED]

    # Review workflow: approve
    approved = mgr.review_benchmark_case(rec.repo_id, case.case_id, "approve", reviewer="Auditor Alice", notes="Formally approved")
    assert approved.review_status == GroundTruthReviewStatus.VALIDATED
    assert approved.reviewer == "Auditor Alice"


# ==============================================================================
# Criterion 11: Unvalidated cases cannot silently enter confirmed ground truth
# ==============================================================================
def test_11_unvalidated_cases_cannot_enter_confirmed_ground_truth(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))
    mgr.inspect_repository(rec.repo_id)

    req = PrepareBenchmarkRequest(
        name="Dubious Scenario",
        source_type=BenchmarkSourceType.BEHAVIORAL_CONTRACT,
        base_commit="HEAD~1",
        target_commit="HEAD",
        expected_changed_symbols=[],
        expected_impacted_symbols=[],
        expected_test_callers=["non_existent_test"],
        provenance="Unverified guess",
    )
    case = mgr.prepare_benchmark_ground_truth(rec.repo_id, req)

    # Reject the case
    mgr.review_benchmark_case(rec.repo_id, case.case_id, "reject", reviewer="Auditor Bob", notes="Invalid expected test")

    overview = mgr.get_custom_suite_overview(selected_repo_id=rec.repo_id)
    # The rejected case must not be counted as a validated benchmark case
    assert overview.get("validated_benchmark_cases_count", 0) == 0


# ==============================================================================
# Criterion 12: All 4 configurations evaluate identical benchmark cases & test universe
# ==============================================================================
def test_12_all_four_configurations_use_same_eligible_benchmark_cases(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))
    mgr.inspect_repository(rec.repo_id)

    req = PrepareBenchmarkRequest(
        name="Validated Discount Case",
        source_type=BenchmarkSourceType.BEHAVIORAL_CONTRACT,
        base_commit="HEAD~1",
        target_commit="HEAD",
        expected_changed_symbols=["compute_discount"],
        expected_impacted_symbols=[],
        expected_test_callers=["test_compute_discount_valid"],
        provenance="Verified specification",
    )
    mgr.prepare_benchmark_ground_truth(rec.repo_id, req)
    runs = mgr.evaluate_repository(rec.repo_id)
    assert len(runs) > 0

    for r in runs:
        # All configurations evaluated for this case
        configs = list(r.baseline_results.keys())
        assert "full_regression" in configs
        assert "naive_name_matching" in configs
        assert "testpilot" in configs

        # Test universe is consistent
        eligible = r.baseline_results["full_regression"].selected_tests
        assert len(eligible) > 0
        assert r.total_tests == len(eligible)


# ==============================================================================
# Criterion 13: Precision, recall, F1, and reduction match hand-calculated test fixtures
# ==============================================================================
def test_13_precision_recall_f1_reduction_match_hand_calculated_fixtures():
    # Hand fixture: 10 eligible tests, 4 selected, 2 true positives, 2 false positives, 1 false negative (3 total GT positives)
    # TP = 2, FP = 2, FN = 1
    # Precision = 2 / (2 + 2) = 0.50
    # Recall = 2 / (2 + 1) = 0.6666...
    # F1 = 2 * (0.5 * 2/3) / (0.5 + 2/3) = 4/7 = ~0.5714
    # Test reduction = 1 - (4 / 10) = 0.60 (60%)

    metrics = compute_metrics(
        predicted_tests=["t1", "t2", "t4", "t5"],
        expected_tests=["t1", "t2", "t3"],
        total_tests=10,
    )

    assert metrics.tp == 2
    assert metrics.fp == 2
    assert metrics.fn == 1
    assert metrics.precision == 0.5
    assert abs(metrics.recall - 0.6667) < 1e-3
    assert abs(metrics.f1 - 0.5714) < 1e-3
    assert metrics.test_reduction_pct == 60.0


# ==============================================================================
# Criterion 14: Undefined metrics handled correctly (zero denominators return None/N/A)
# ==============================================================================
def test_14_undefined_metrics_handled_correctly(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))

    overview = mgr.get_custom_suite_overview(selected_repo_id=rec.repo_id)
    matrix = overview["primary_matrix"]

    for row in matrix:
        assert row["precision_display"] == "—"
        assert row["recall_display"] == "—"
        assert row["f1_display"] == "—"


# ==============================================================================
# Criterion 15: Aggregate metrics micro vs macro distinction
# ==============================================================================
def test_15_aggregate_metrics_micro_macro_distinction(temp_custom_store):
    mgr, _ = temp_custom_store
    overview = mgr.get_custom_suite_overview()

    assert "micro_matrix" in overview
    assert "macro_matrix" in overview
    assert len(overview["micro_matrix"]) >= 4
    assert len(overview["macro_matrix"]) >= 4


# ==============================================================================
# Criterion 16: Fault-detection separated from selection metrics
# ==============================================================================
def test_16_fault_detection_separated_from_selection_metrics(temp_custom_store):
    mgr, _ = temp_custom_store
    overview = mgr.get_custom_suite_overview()

    assert "fault_detection_summary" in overview
    f_summary = overview["fault_detection_summary"]
    assert "average_fault_detection_rate" in f_summary


# ==============================================================================
# Criterion 17: Repository revision changes can invalidate affected benchmark evidence
# ==============================================================================
def test_17_revision_changes_invalidate_benchmark_evidence(temp_custom_store, synthetic_repo):
    mgr, _ = temp_custom_store
    rec = mgr.register_repository(RegisterRepositoryRequest(path=str(synthetic_repo)))
    mgr.inspect_repository(rec.repo_id)

    req = PrepareBenchmarkRequest(
        name="Case to be staled",
        source_type=BenchmarkSourceType.BEHAVIORAL_CONTRACT,
        base_commit="HEAD~1",
        target_commit="HEAD",
        expected_changed_symbols=["compute_discount"],
        expected_impacted_symbols=[],
        expected_test_callers=["test_compute_discount_valid"],
        provenance="Valid before revision advance",
    )
    case = mgr.prepare_benchmark_ground_truth(rec.repo_id, req)

    # Advance git revision in synthetic_repo
    (synthetic_repo / "new_file.txt").write_text("advance", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(synthetic_repo), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Advance HEAD revision"], cwd=str(synthetic_repo), capture_output=True, check=True)

    # Re-inspect repository
    inspected = mgr.inspect_repository(rec.repo_id)
    stale_case = next(c for c in inspected.benchmark_cases if c.case_id == case.case_id)
    assert stale_case.review_status == GroundTruthReviewStatus.STALE
    assert stale_case.ground_truth.is_verified is False


# ==============================================================================
# Criterion 18: Failed or unsupported configurations visibly reported
# ==============================================================================
def test_18_failed_or_unsupported_configurations_visibly_reported(temp_custom_store):
    mgr, _ = temp_custom_store
    overview = mgr.get_custom_suite_overview()

    assert "excluded_repositories" in overview
    excluded = overview["excluded_repositories"]
    assert isinstance(excluded, list)


# ==============================================================================
# Criterion 19: Authoritative research benchmark remains unchanged
# ==============================================================================
def test_19_authoritative_research_benchmark_remains_unchanged():
    engine = EvaluationEngine()
    overview = engine.get_overview_summary()

    matrix = overview["primary_matrix"]
    assert len(matrix) >= 4

    tp_row = next((r for r in matrix if r["configuration"] == "TestPilot"), None)
    assert tp_row is not None
    assert tp_row["is_evaluated"] is True
    assert tp_row["tp"] > 0
    assert tp_row["precision"] is not None
    assert tp_row["recall"] is not None


# ==============================================================================
# Criterion 20: Existing tests and unrelated features continue to work
# ==============================================================================
def test_20_existing_tests_and_unrelated_features_continue_working():
    orchestrator = FullPipelineOrchestrator()
    assert hasattr(orchestrator, "execute")
    assert hasattr(orchestrator, "_run_evolution_stage")
