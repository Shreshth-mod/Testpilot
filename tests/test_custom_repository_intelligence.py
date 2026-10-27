"""
Tests for Custom Repository Intelligence Platform:
- Guided Onboarding (Public Git, Private Git with token security, Local Folders, Pre-Flight Validation)
- Whole-Repository Intelligence & Durable Generation Pipeline (Explicit Lifecycle States, Persistence, Crash Recovery, Anti-Trivial Checks)
- Historical Bug-Fix Ground Truth Mining & Dual-Revision Verification
- FinMath Test Selection Defect Fix Verification (TP=3, FN=0, Recall=1.0)
"""

import os
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from testpilot.evaluation.auto_benchmark import AutoBenchmarkGenerator
from testpilot.evaluation.bug_mining import HistoricalBugMiner
from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    CustomRepositoryRecord,
    GeneratedTestRecord,
    GroundTruthReviewStatus,
    RepositoryStatus,
)
from testpilot.evaluation.generation_manager import GenerationJobManager
from testpilot.evaluation.generation_models import (
    GenerationJobRecord,
    GenerationJobStatus,
    HistoricalBugCandidate,
    RepositoryValidationSummary,
)
from testpilot.sourcegraph.client import LocalCodeGraphFallback


@pytest.fixture
def temp_storage():
    tmp = tempfile.mkdtemp(prefix="testpilot_intel_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_local_repo(temp_storage):
    """Creates a realistic minimal Python repository with implementation and tests."""
    repo_dir = temp_storage / "sample_math_repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "src").mkdir()
    (repo_dir / "tests").mkdir()

    # Implementation
    (repo_dir / "src" / "math_ops.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    # Existing test
    (repo_dir / "tests" / "test_math.py").write_text(
        "from src.math_ops import add, subtract\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_subtract():\n"
        "    assert subtract(10, 4) == 6\n",
        encoding="utf-8",
    )

    return repo_dir


# =============================================================================
# 1. Onboarding & Pre-Registration Validation Tests
# =============================================================================


def test_validate_local_folder_candidate(temp_storage, sample_local_repo):
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(temp_storage))
    summary = mgr.validate_repository_candidate(str(sample_local_repo))

    assert isinstance(summary, RepositoryValidationSummary)
    assert summary.is_valid is True
    assert summary.source_type == "local_folder"
    assert summary.repo_name == "sample_math_repo"
    assert summary.existing_tests_count >= 2
    assert "python" in [l.lower() for l in summary.languages]
    assert summary.test_framework == "pytest"
    assert summary.private_access_status == "not_required"


def test_validate_nonexistent_local_folder(temp_storage):
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(temp_storage))
    summary = mgr.validate_repository_candidate(str(temp_storage / "nonexistent_dir_12345"))

    assert summary.is_valid is False
    assert "does not exist" in (summary.error_message or "").lower()


def test_validate_public_git_candidate(temp_storage):
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(temp_storage))

    # Mock subprocess.run for git ls-remote to simulate reachable public repo
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ref: refs/heads/main\tHEAD\n3c7bec58\tHEAD\n", stderr="")
        summary = mgr.validate_repository_candidate("https://github.com/Shreshth-mod/FinMath")

        assert summary.is_valid is True
        assert summary.source_type == "public_git"
        assert summary.repo_name == "FinMath"
        assert summary.branch == "main"
        assert summary.private_access_status == "not_required"


def test_validate_private_git_candidate_with_auth(temp_storage):
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(temp_storage))

    # Mock git ls-remote with token
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ref: refs/heads/main\tHEAD\n", stderr="")
        summary = mgr.validate_repository_candidate(
            "https://github.com/org/super-secret-repo",
            auth_token="ghp_test_secret_token_1234567890",
        )

        assert summary.is_valid is True
        assert summary.source_type == "private_git"
        assert summary.repo_name == "super-secret-repo"
        assert summary.private_access_status == "authenticated"

        # Verify token was passed in http.extraHeader and NOT in URL
        cmd_called = mock_run.call_args[0][0]
        assert not any("ghp_test_secret_token_1234567890" in arg for arg in cmd_called if "http.extraHeader" not in arg)


def test_validate_private_git_auth_failure_sanitization(temp_storage):
    mgr = CustomRepositoryBenchmarkManager(storage_dir=str(temp_storage))

    # Simulate git auth failure with token
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout="",
            stderr="fatal: Authentication failed for 'https://github.com/org/private-repo.git/' with token ghp_sensitive_secret_9999",
        )
        summary = mgr.validate_repository_candidate(
            "https://github.com/org/private-repo",
            auth_token="ghp_sensitive_secret_9999",
        )

        assert summary.is_valid is False
        assert summary.private_access_status == "auth_failed"
        # Verify secret token was redacted from error message
        assert "ghp_sensitive_secret_9999" not in (summary.error_message or "")


# =============================================================================
# 2. Durable Test Generation Job Manager & Crash Recovery Tests
# =============================================================================


def test_generation_job_creation_and_persistence(temp_storage):
    job_mgr = GenerationJobManager(storage_dir=str(temp_storage))
    job = job_mgr.create_job("repo_test_123", commit_sha="abc1234")

    assert job.job_id.startswith("gen_repo_test_123")
    assert job.status == GenerationJobStatus.QUEUED
    assert (job_mgr.jobs_dir / f"{job.job_id}.json").exists()

    # Re-instantiate manager to verify persistence across restarts
    job_mgr_reloaded = GenerationJobManager(storage_dir=str(temp_storage))
    loaded_job = job_mgr_reloaded.get_job(job.job_id)
    assert loaded_job is not None
    # Queued job should have been recovered on startup
    assert loaded_job.status in (GenerationJobStatus.FAILED, GenerationJobStatus.COMPLETED_WITH_WARNINGS)
    assert "Recovered" in (loaded_job.failure_reason or "")


def test_generation_job_cancellation(temp_storage):
    job_mgr = GenerationJobManager(storage_dir=str(temp_storage))
    job = job_mgr.create_job("repo_cancel_test")

    # Manually mark as analyzing to test in-flight cancellation
    job.status = GenerationJobStatus.ANALYZING
    job_mgr.save_job(job)

    cancelled = job_mgr.cancel_job(job.job_id)
    assert cancelled is True

    updated = job_mgr.get_job(job.job_id)
    assert updated.status == GenerationJobStatus.CANCELLED
    assert updated.completed_at is not None

    # Cancelling already cancelled or terminal job returns False
    assert job_mgr.cancel_job(job.job_id) is False


def test_anti_trivial_test_rejection(temp_storage):
    test_dir = temp_storage / "anti_trivial_repo"
    test_dir.mkdir(parents=True)
    tests_sub = test_dir / "tests" / "generated"
    tests_sub.mkdir(parents=True)

    test_file = tests_sub / "test_math.py"
    test_file.write_text(
        "def test_trivial():\n"
        "    assert True\n\n"
        "def test_good():\n"
        "    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    generator = AutoBenchmarkGenerator(repo_path=test_dir)
    records = [
        GeneratedTestRecord(
            test_id="t_trivial",
            target_symbol="add",
            target_file="src/math.py",
            test_file=str(test_file),
            test_name="test_trivial",
            intended_behavior="Verify add returns sum",
            test_code="def test_trivial():\n    assert True\n",
        ),
        GeneratedTestRecord(
            test_id="t_good",
            target_symbol="add",
            target_file="src/math.py",
            test_file=str(test_file),
            test_name="test_good",
            intended_behavior="Verify add arithmetic",
            test_code="def test_good():\n    assert 2 + 2 == 4\n",
        ),
    ]

    val_records = generator.validate_generated_tests(records)
    rec_trivial = next(r for r in val_records if r.test_id == "t_trivial")
    assert rec_trivial.is_trivial is True
    assert "trivial" in (rec_trivial.rejection_reason or "").lower()


# =============================================================================
# 3. Historical Bug-Fix Ground Truth Mining Tests
# =============================================================================


def test_historical_bug_miner_discovery(temp_storage):
    repo_dir = temp_storage / "git_mock_repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()

    miner = HistoricalBugMiner(repo_path=repo_dir)

    mock_git_log = (
        "b1c2d3e4f5|a0b1c2d3e4|Fix ZeroDivisionError in calculation (#42)\n"
        "c2d3e4f5a6|b1c2d3e4f5|Refactor documentation and comments\n"
    )

    with patch("subprocess.run") as mock_run:
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if "log" in cmd_str:
                return MagicMock(returncode=0, stdout=mock_git_log, stderr="")
            if "diff-tree" in cmd_str:
                return MagicMock(returncode=0, stdout="src/calc.py\ntests/test_calc.py\n", stderr="")
            if "show" in cmd_str:
                return MagicMock(returncode=0, stdout="def test_calc_zero_div():\n    assert True\n", stderr="")
            if "symbol" in cmd_str or "diff" in cmd_str:
                return MagicMock(returncode=0, stdout="calc_formula\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        candidates = miner.mine_candidate_bug_fixes("repo_mock_id", max_commits=10, max_candidates=5)

        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.commit_sha == "b1c2d3e4f5"
        assert cand.parent_commit_sha == "a0b1c2d3e4"
        assert "ZeroDivisionError" in cand.commit_message
        assert "tests/test_calc.py" in cand.test_files
        assert any("test_calc_zero_div" in t for t in cand.candidate_tests)


def test_historical_bug_dual_revision_validation(temp_storage):
    repo_dir = temp_storage / "git_mock_repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()

    miner = HistoricalBugMiner(repo_path=repo_dir)

    candidate = HistoricalBugCandidate(
        candidate_id="cand_test_01",
        repo_id="repo_mock_id",
        commit_sha="b1c2d3e4f5",
        parent_commit_sha="a0b1c2d3e4",
        commit_message="Fix overflow in add()",
        changed_files=["src/math.py"],
        changed_symbols=["add"],
        test_files=["tests/test_math.py"],
        candidate_tests=["tests/test_math.py::test_add_overflow"],
    )

    # Validate that validate_bug_candidate marks status correctly when parent fails and fixed commit passes
    with patch.object(miner, "_run_tests_in_workspace") as mock_run_test:
        # First call is buggy parent (returns has_failure=True, all_passed=False)
        # Second call is fixed child (returns has_failure=False, all_passed=True)
        mock_run_test.side_effect = [
            {"summary": "FAILED (1 error)", "has_failure": True, "all_passed": False},
            {"summary": "PASSED (1 test)", "has_failure": False, "all_passed": True},
        ]
        with patch.object(miner, "_export_archive"):
            with patch.object(miner, "_overlay_tests_from_commit"):
                validated_cand = miner.validate_bug_candidate(candidate)

                assert validated_cand.validation_status == GroundTruthReviewStatus.VALIDATED
                assert validated_cand.before_result == "FAILED (1 error)"
                assert validated_cand.after_result == "PASSED (1 test)"

                case = miner.convert_candidate_to_benchmark_case(validated_cand)
                assert isinstance(case, CustomBenchmarkCase)
                assert case.ground_truth.is_verified is True
                assert any("test_add_overflow" in t for t in case.ground_truth.expected_test_callers)


# =============================================================================
# 4. FinMath Defect Fix Verification (Module Path Truncation Fix)
# =============================================================================


def test_finmath_module_matches_path_fix():
    """
    Direct regression test for the FinMath defect:
    In Python, 'app.py'.rstrip('.py') returns 'a' because 'p' and '.' are stripped.
    Verify that _module_matches_path properly resolves 'app' against 'app.py' and 'src/app.py'.
    """
    norm_path = "app.py"
    if norm_path.endswith(".py"):
        stem = norm_path[:-3]
    assert stem == "app", f"Expected 'app', got '{stem}'"

    fallback = LocalCodeGraphFallback()
    assert fallback._module_matches_path("app", "app.py") is True
    assert fallback._module_matches_path("app", "src/app.py") is True
    assert fallback._module_matches_path("math_ops", "src/math_ops.py") is True
    assert fallback._module_matches_path("wrong", "app.py") is False


def test_finmath_repository_evaluation_matrix_reproduction():
    """
    Evaluates FinMath repository in custom_repositories.json with our AST module fix.
    Ensures that TestPilot achieves TP=3, FN=0, Recall=1.0.
    """
    repo_file = Path("testbed/evaluation_store/custom_repositories.json")
    if not repo_file.exists():
        pytest.skip("FinMath custom repository benchmark store not found.")

    mgr = CustomRepositoryBenchmarkManager()
    finmath_id = "repo_shreshth-mod_finmath_db2b8f524f"
    rec = mgr.get_repository(finmath_id)
    if not rec or not rec.benchmark_cases:
        pytest.skip(f"FinMath repo {finmath_id} not registered with benchmark cases.")

    runs = mgr.evaluate_repository(finmath_id)
    assert len(runs) > 0, "No evaluation runs returned for FinMath"
    run = runs[0]

    assert "testpilot" in run.metrics, "TestPilot missing from FinMath run metrics"
    tp_metric = run.metrics["testpilot"]

    assert tp_metric.tp == 3, f"Expected TP=3 for FinMath, got {tp_metric.tp}"
    assert tp_metric.fn == 0, f"Expected FN=0 for FinMath, got {tp_metric.fn}"
    assert tp_metric.recall == 1.0, f"Expected Recall=1.0, got {tp_metric.recall}"
    assert tp_metric.precision == 1.0, f"Expected Precision=1.0, got {tp_metric.precision}"
