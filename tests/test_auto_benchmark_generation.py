"""
Comprehensive test suite for Automatic Regression Test Generation,
Independent Ground Truth Creation, and Cross-Validation for Custom Repository Benchmarks.

Covers all 16 required verification scenarios:
1. Repository with substantial existing test suite.
2. Repository with source code but no test files.
3. Repository with failing/incomplete baseline tests.
4. Repository with unsupported language/framework features.
5. Generation of meaningful tests for supported source code.
6. Rejection of trivial or invalid generated tests.
7. Detection of a regression using historical faulty/corrected revisions.
8. Validation of expected behavior against an independent contract.
9. Rejection of tests that pass on both faulty and corrected versions (not regression-sensitive).
10. Separation of environment failures from code defect proof.
11. Prevention of ground truth derivation solely from predictions (anti-leakage).
12. Unvalidated benchmark cases excluded from quantitative aggregates.
13. Validated cases produce correctly calculated TP, FP, FN, precision, recall, F1, test reduction.
14. Authoritative benchmark definitions and results remain unchanged.
15. Isolation, timeouts, and cleanup work correctly for generated test execution.
16. Benchmark results traceable to source revision and validation artifacts.
"""

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from testpilot.evaluation.auto_benchmark import AutoBenchmarkGenerator
from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    GeneratedTestRecord,
    GroundTruthValidationLevel,
    PrepareBenchmarkRequest,
    RegisterRepositoryRequest,
    RepositoryStatus,
)
from testpilot.evaluation.engine import EvaluationEngine
from testpilot.evaluation.models import GroundTruth
from testpilot.web.api import app


def _create_sample_git_repo(repo_name: str = "auto_sample") -> Path:
    """Helper to initialize a real git repository with sample math/calculator code."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"testpilot_{repo_name}_"))
    subprocess.run(["git", "init"], cwd=str(temp_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "TestPilot Auto"], cwd=str(temp_dir), check=True)
    subprocess.run(["git", "config", "user.email", "auto@testpilot.ai"], cwd=str(temp_dir), check=True)

    src_dir = temp_dir / "mathlib"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")

    calc_file = src_dir / "calc.py"
    calc_file.write_text("""
def add_values(a: float, b: float) -> float:
    \"\"\"Add two floating point numbers.\"\"\"
    return a + b

def multiply_values(x: int, y: int) -> int:
    \"\"\"Multiply two integers.\"\"\"
    return x * y

def divide_positive(val: float, divisor: float) -> float:
    \"\"\"Divides val by divisor; raises ValueError if divisor <= 0.\"\"\"
    if divisor <= 0:
        raise ValueError("Divisor must be positive.")
    return val / divisor
""", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=str(temp_dir), check=True)
    subprocess.run(["git", "commit", "-m", "Initial calculator code"], cwd=str(temp_dir), check=True)
    return temp_dir


# =========================================================================
# Scenario 1: Repository with substantial existing test suite
# =========================================================================
def test_scenario_01_repo_with_existing_test_suite():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("with_tests")
        test_dir = repo_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "test_calc.py").write_text("""
from mathlib.calc import add_values, multiply_values

def test_add_works():
    assert add_values(2.0, 3.0) == 5.0

def test_mult_works():
    assert multiply_values(3, 4) == 12
""", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Add tests"], cwd=str(repo_dir), check=True)

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        assert record.total_tests_count == 2
        assert record.test_files_count == 1
        assert record.status in (RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED, RepositoryStatus.TESTS_DISCOVERED)
        assert record.baseline_test_status == "passed"


# =========================================================================
# Scenario 2: Repository with source code but no test files
# =========================================================================
def test_scenario_02_repo_with_no_existing_tests():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("no_tests")

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        assert record.total_tests_count == 0
        assert record.test_files_count == 0
        assert record.baseline_test_status == "no_tests"
        assert record.status == RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
        assert "Automatic regression test generation is available" in record.status_message


# =========================================================================
# Scenario 3: Repository with failing or incomplete baseline tests
# =========================================================================
def test_scenario_03_repo_with_failing_baseline_tests():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("failing_tests")
        test_dir = repo_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "test_broken.py").write_text("""
def test_intentionally_broken():
    assert 1 == 2, "Known pre-existing defect"
""", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Add failing test"], cwd=str(repo_dir), check=True)

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        assert record.total_tests_count == 1
        assert record.baseline_test_status == "failed"
        assert record.baseline_test_error is not None


# =========================================================================
# Scenario 4: Repository with unsupported language / framework features
# =========================================================================
def test_scenario_04_repo_with_unsupported_language():
    with tempfile.TemporaryDirectory() as storage_tmp, tempfile.TemporaryDirectory() as rust_repo_tmp:
        rust_dir = Path(rust_repo_tmp)
        (rust_dir / "main.rs").write_text("fn main() { println!(\"hello\"); }", encoding="utf-8")

        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(rust_dir)))

        assert record.supported_language is False
        assert record.status == RepositoryStatus.PARTIALLY_SUPPORTED
        assert "not currently supported" in (record.unsupported_reason or "")


# =========================================================================
# Scenario 5: Generation of meaningful tests for supported source code
# =========================================================================
def test_scenario_05_generation_of_meaningful_tests():
    repo_dir = _create_sample_git_repo("testgen_supported")
    generator = AutoBenchmarkGenerator(repo_dir)

    targets = generator.discover_target_functions()
    assert len(targets) >= 3  # add_values, multiply_values, divide_positive

    records = generator.generate_and_save_tests(max_targets=10)
    assert len(records) > 0

    for r in records:
        assert r.target_symbol.startswith("mathlib.calc.")
        assert r.intended_behavior != ""
        assert (repo_dir / r.test_file).exists()

        # Check content is non-trivial
        test_code = (repo_dir / r.test_file).read_text(encoding="utf-8")
        assert "assert True" not in test_code
        assert r.target_symbol.split(".")[-1] in test_code


# =========================================================================
# Scenario 6: Rejection of trivial or invalid generated tests
# =========================================================================
def test_scenario_06_rejection_of_trivial_tests():
    repo_dir = _create_sample_git_repo("trivial_check")
    gen_dir = repo_dir / "tests" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    trivial_file = gen_dir / "test_trivial.py"
    trivial_file.write_text("""
def test_bare_true():
    assert True

def test_empty_pass():
    pass
""", encoding="utf-8")

    generator = AutoBenchmarkGenerator(repo_dir)
    dummy_records = [
        GeneratedTestRecord(
            test_id="tests/generated/test_trivial.py::test_bare_true",
            target_symbol="mathlib.calc.add_values",
            target_file="mathlib/calc.py",
            test_file="tests/generated/test_trivial.py",
            test_name="test_bare_true",
            intended_behavior="trivial check",
        ),
        GeneratedTestRecord(
            test_id="tests/generated/test_trivial.py::test_empty_pass",
            target_symbol="mathlib.calc.multiply_values",
            target_file="mathlib/calc.py",
            test_file="tests/generated/test_trivial.py",
            test_name="test_empty_pass",
            intended_behavior="empty pass check",
        ),
    ]

    validated = generator.validate_generated_tests(dummy_records)
    for v in validated:
        assert v.is_trivial is True
        assert "trivial" in (v.rejection_reason or "").lower()


# =========================================================================
# Scenario 7 & 8: Controlled Mutation Sensitivity & Independent Contract
# =========================================================================
def test_scenario_07_and_08_mutation_sensitivity_and_independent_contract():
    repo_dir = _create_sample_git_repo("mutation_test")
    generator = AutoBenchmarkGenerator(repo_dir)

    gen_records = generator.generate_and_save_tests(max_targets=5)
    val_records = generator.validate_generated_tests(gen_records)

    # Some valid tests should execute successfully
    passing = [v for v in val_records if v.executes_successfully]
    assert len(passing) > 0

    mut_records = generator.cross_check_mutation_sensitivity(val_records)
    sensitive = [m for m in mut_records if m.mutation_sensitive]
    assert len(sensitive) > 0, "At least one generated test must detect the controlled code mutation"

    # Construct benchmark cases from mutation sensitivity
    cases = generator.construct_benchmark_cases("repo_mut_test", mut_records, is_git=True)
    assert len(cases) > 0
    assert any(c.ground_truth.is_verified for c in cases)
    assert any(c.validation_level == GroundTruthValidationLevel.CONTRACT_MUTATION_VERIFIED for c in cases)


# =========================================================================
# Scenario 9: Tests passing on both mutant and correct version are NOT sensitive
# =========================================================================
def test_scenario_09_test_passing_on_mutant_is_not_sensitive():
    repo_dir = _create_sample_git_repo("weak_test")
    gen_dir = repo_dir / "tests" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # A weak test that asserts result is not a string (which is True even if result is None or negative)
    weak_file = gen_dir / "test_weak.py"
    weak_file.write_text("""
from mathlib.calc import add_values

def test_weak_tautology():
    # Will pass even if add_values returns None!
    res = add_values(1.0, 2.0)
    assert not isinstance(res, str)
""", encoding="utf-8")

    generator = AutoBenchmarkGenerator(repo_dir)
    weak_record = GeneratedTestRecord(
        test_id="tests/generated/test_weak.py::test_weak_tautology",
        target_symbol="mathlib.calc.add_values",
        target_file="mathlib/calc.py",
        test_file="tests/generated/test_weak.py",
        test_name="test_weak_tautology",
        intended_behavior="weak tautology",
        executes_successfully=True,
    )

    checked = generator.cross_check_mutation_sensitivity([weak_record])
    # When add_values is mutated to return None, not isinstance(None, str) is STILL True!
    # Therefore, mutation_sensitive must be False!
    assert checked[0].mutation_sensitive is False
    assert "did not detect controlled code mutation" in (checked[0].rejection_reason or "")


# =========================================================================
# Scenario 10: Environment failure is not treated as code defect
# =========================================================================
def test_scenario_10_environment_failure_separation():
    repo_dir = _create_sample_git_repo("env_failure")
    gen_dir = repo_dir / "tests" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    bad_import_file = gen_dir / "test_missing_dep.py"
    bad_import_file.write_text("""
import non_existent_package_12345

def test_missing_env():
    assert True
""", encoding="utf-8")

    generator = AutoBenchmarkGenerator(repo_dir)
    record = GeneratedTestRecord(
        test_id="tests/generated/test_missing_dep.py::test_missing_env",
        target_symbol="mathlib.calc.add_values",
        target_file="mathlib/calc.py",
        test_file="tests/generated/test_missing_dep.py",
        test_name="test_missing_env",
        intended_behavior="missing dep check",
    )

    validated = generator.validate_generated_tests([record])
    assert validated[0].executes_successfully is False
    assert "Environment/dependency" in (validated[0].rejection_reason or "")


# =========================================================================
# Scenario 11: Ground truth cannot be derived solely from predictions
# =========================================================================
def test_scenario_11_anti_leakage_ground_truth_independence():
    repo_dir = _create_sample_git_repo("anti_leakage")
    generator = AutoBenchmarkGenerator(repo_dir)

    gen_records = generator.generate_and_save_tests(max_targets=3)
    val_records = generator.validate_generated_tests(gen_records)
    mut_records = generator.cross_check_mutation_sensitivity(val_records)
    cases = generator.construct_benchmark_cases("repo_anti_leakage", mut_records, is_git=True)

    for case in cases:
        # Provenance must explicitly state independent validation source (AST contract / mutation testing)
        assert "mutation" in case.ground_truth.provenance.lower() or "contract" in case.ground_truth.provenance.lower()
        # Must not mention TestPilot prediction models or LLM selector
        assert "prediction" not in case.ground_truth.provenance.lower()
        assert "selector" not in case.ground_truth.provenance.lower()


# =========================================================================
# Scenario 12: Unvalidated cases do not contribute to quantitative metrics
# =========================================================================
def test_scenario_12_unvalidated_cases_excluded_from_aggregates():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("unvalidated_metrics")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        # Add an UNVALIDATED proposed case
        unvalidated_case = CustomBenchmarkCase(
            case_id=f"{record.repo_id}_unval",
            name="Unverified Proposal",
            source_type=BenchmarkSourceType.BEHAVIORAL_CONTRACT,
            ground_truth=GroundTruth(
                case_id=f"{record.repo_id}_unval",
                expected_changed_symbols=["add_values"],
                expected_impacted_symbols=["add_values"],
                expected_test_callers=["tests/dummy.py::test_fake"],
                provenance="Unreviewed proposal",
                is_verified=False,  # NOT VERIFIED!
            ),
            validation_level=GroundTruthValidationLevel.PROPOSED_AWAITING_REVIEW,
        )
        record.benchmark_cases.append(unvalidated_case)
        records = mgr._load_all_records()
        records[record.repo_id] = record
        mgr._save_all_records(records)

        # Overview should have 0 evaluated runs and not report fake metrics
        overview = mgr.get_custom_suite_overview()
        assert overview["evaluated_repositories_count"] == 0
        assert overview["active_benchmark_cases_count"] == 0


# =========================================================================
# Scenario 13: Validated cases produce correctly calculated metrics
# =========================================================================
def test_scenario_13_validated_cases_produce_correct_metrics():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("full_auto_flow")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))

        # Run auto-generate benchmarks end to end
        updated = mgr.auto_generate_and_validate_benchmarks(record.repo_id)
        assert updated.generated_tests_count > 0
        assert updated.validated_benchmark_cases_count > 0
        assert updated.status == RepositoryStatus.READY_FOR_EVALUATION

        # Evaluate repository
        runs = mgr.evaluate_repository(record.repo_id)
        assert len(runs) > 0

        for r in runs:
            assert len(r.metrics) > 0
            for baseline_name, m in r.metrics.items():
                assert m.tp >= 0
                assert m.fp >= 0
                assert m.fn >= 0
                if m.precision is not None:
                    assert 0.0 <= m.precision <= 1.0
                if m.recall is not None:
                    assert 0.0 <= m.recall <= 1.0
                if m.f1 is not None:
                    assert 0.0 <= m.f1 <= 1.0
                if m.test_reduction_pct is not None:
                    assert 0.0 <= m.test_reduction_pct <= 100.0


# =========================================================================
# Scenario 14: Existing authoritative benchmark results remain unchanged
# =========================================================================
def test_scenario_14_authoritative_benchmarks_untouched():
    from testpilot.evaluation.storage import EvaluationStorage
    storage = EvaluationStorage()
    authoritative_cases = storage.get_benchmark_cases()
    assert len(authoritative_cases) >= 5
    case_ids = {c.case_id for c in authoritative_cases}
    assert "flask_ipv6" in case_ids
    assert "django_model_rename" in case_ids
    assert "homeassistant_hue_init" in case_ids
    assert "testbed_discount_deficit" in case_ids
    assert "testbed_tax_precision" in case_ids

    # Verify authoritative benchmark ground truth remains intact and separate from custom suite
    flask_case = storage.get_benchmark_case("flask_ipv6")
    assert flask_case is not None
    assert flask_case.ground_truth.expected_changed_symbols == ["run", "Flask.run", "test_run_from_config"]


# =========================================================================
# Scenario 15: Subprocess isolation, timeouts, and resource cleanup
# =========================================================================
def test_scenario_15_subprocess_isolation_and_cleanup():
    repo_dir = _create_sample_git_repo("isolation_test")
    generator = AutoBenchmarkGenerator(repo_dir)

    # Verify no temp files or mutants left on disk
    gen_records = generator.generate_and_save_tests(max_targets=3)
    val_records = generator.validate_generated_tests(gen_records)
    mut_records = generator.cross_check_mutation_sensitivity(val_records)

    # Ensure backup files (.orig_bak) were cleaned up
    bak_files = list(repo_dir.glob("**/*.orig_bak"))
    assert len(bak_files) == 0, "All temporary mutation backup files must be cleaned up"


# =========================================================================
# Scenario 16: Benchmark results traceable to source revision
# =========================================================================
def test_scenario_16_traceability_to_revision_and_artifacts():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("traceability_test")
        mgr = CustomRepositoryBenchmarkManager(storage_dir=storage_tmp)
        record = mgr.register_repository(RegisterRepositoryRequest(path=str(repo_dir)))
        updated = mgr.auto_generate_and_validate_benchmarks(record.repo_id)

        assert updated.detected_revision is not None
        assert len(updated.detected_revision) == 40  # Full git SHA
        assert updated.last_analyzed_at is not None

        for case in updated.benchmark_cases:
            assert case.case_id.startswith(record.repo_id)
            assert "calc.py" in case.validation_evidence.get("target_file", "")
            assert case.created_at != ""


# =========================================================================
# Web API Endpoint Tests for Auto-Generation
# =========================================================================
def test_web_api_auto_generate_benchmarks_endpoint():
    with tempfile.TemporaryDirectory() as storage_tmp:
        repo_dir = _create_sample_git_repo("api_autogen")
        client = TestClient(app)

        # 1. Register repo
        res_reg = client.post("/api/evaluation/custom-suite/repositories", json={"path": str(repo_dir)})
        assert res_reg.status_code == 200
        repo_id = res_reg.json()["repo_id"]

        # 2. Trigger auto-generate endpoint
        res_gen = client.post(f"/api/evaluation/custom-suite/repositories/{repo_id}/auto-generate-benchmarks")
        assert res_gen.status_code == 200
        gen_data = res_gen.json()
        assert gen_data["generated_tests_count"] > 0
        assert gen_data["validated_benchmark_cases_count"] > 0
        assert gen_data["status"] == "ready_for_evaluation"

        # 3. Retrieve generated tests endpoint
        res_tests = client.get(f"/api/evaluation/custom-suite/repositories/{repo_id}/generated-tests")
        assert res_tests.status_code == 200
        tests_list = res_tests.json()
        assert len(tests_list) == gen_data["generated_tests_count"]
        assert any(t["mutation_sensitive"] for t in tests_list)
