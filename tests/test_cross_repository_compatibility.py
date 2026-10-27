"""
Test Suite for Cross-Repository Compatibility and Capability Detection Subsystem.
Verifies multi-ecosystem detection (Python, TypeScript/JavaScript, Go, Monorepos),
per-stage capability classifications, missing external services, and refusal of unsupported operations.
"""

from pathlib import Path
import pytest
import tempfile
import shutil

from testpilot.capabilities import (
    CapabilityDetector,
    PipelineStage,
    RepositoryCompatibilityReport,
    SupportLevel,
)
from testpilot.evaluation.baselines import BaselineEvaluator
from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import RegisterRepositoryRequest, RepositoryStatus


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cross_repo"
PYTHON_FIXTURE = FIXTURES_DIR / "python_pkg"
JS_TS_FIXTURE = FIXTURES_DIR / "js_ts_pkg"
MONOREPO_FIXTURE = FIXTURES_DIR / "monorepo_pkg"
SERVICES_FIXTURE = FIXTURES_DIR / "missing_services_pkg"


class TestCrossRepositoryCompatibility:
    """Comprehensive test suite covering cross-repository capability detection."""

    def test_python_ecosystem_capabilities(self):
        """Verifies full capability detection for standard Python pytest package."""
        report = CapabilityDetector.generate_report(PYTHON_FIXTURE)

        assert report.ecosystem.primary_language == "Python"
        assert "Python" in report.ecosystem.detected_languages
        assert "pytest" in report.ecosystem.detected_test_frameworks
        assert len(report.ecosystem.manifests) >= 1
        assert report.ecosystem.manifests[0].manifest_type == "pyproject.toml"
        assert "pydantic" in report.ecosystem.manifests[0].dependencies

        # Per-stage capabilities
        onboarding = report.get_stage(PipelineStage.ONBOARDING)
        assert onboarding.support_level == SupportLevel.FULLY_SUPPORTED
        assert onboarding.available is True

        analysis = report.get_stage(PipelineStage.ANALYSIS)
        assert analysis.support_level == SupportLevel.FULLY_SUPPORTED
        assert analysis.available is True

        discovery = report.get_stage(PipelineStage.TEST_DISCOVERY)
        assert discovery.support_level == SupportLevel.FULLY_SUPPORTED
        assert discovery.available is True

        generation = report.get_stage(PipelineStage.TEST_GENERATION)
        assert generation.support_level == SupportLevel.FULLY_SUPPORTED
        assert generation.available is True

        # Test discovery check
        evaluator = BaselineEvaluator(str(PYTHON_FIXTURE))
        tests = evaluator.discover_all_tests()
        assert len(tests) >= 4
        assert any("test_add" in t for t in tests)
        assert any("test_divide_by_zero" in t for t in tests)

    def test_javascript_typescript_capabilities(self):
        """Verifies honest partial support classification for JavaScript/TypeScript repositories."""
        report = CapabilityDetector.generate_report(JS_TS_FIXTURE)

        assert report.ecosystem.primary_language in ("TypeScript", "JavaScript")
        assert "jest" in report.ecosystem.detected_test_frameworks
        manifest_types = [m.manifest_type for m in report.ecosystem.manifests]
        assert "package.json" in manifest_types
        assert "tsconfig.json" in manifest_types

        # Onboarding: supported
        assert report.get_stage(PipelineStage.ONBOARDING).support_level == SupportLevel.FULLY_SUPPORTED

        # Analysis: partially supported (structural export/function inspection)
        analysis_cap = report.get_stage(PipelineStage.ANALYSIS)
        assert analysis_cap.support_level == SupportLevel.PARTIALLY_SUPPORTED
        assert analysis_cap.available is True
        assert analysis_cap.missing_adapter is not None

        # Test discovery: supported via JS/TS test discovery
        discovery_cap = report.get_stage(PipelineStage.TEST_DISCOVERY)
        assert discovery_cap.support_level == SupportLevel.FULLY_SUPPORTED
        assert discovery_cap.available is True

        # Test generation: UNSUPPORTED with clear missing adapter
        gen_cap = report.get_stage(PipelineStage.TEST_GENERATION)
        assert gen_cap.support_level == SupportLevel.UNSUPPORTED
        assert gen_cap.available is False
        assert "Autonomous test generator" in gen_cap.missing_adapter

        # Docker execution: UNSUPPORTED (missing node runner)
        exec_cap = report.get_stage(PipelineStage.DOCKER_EXECUTION)
        assert exec_cap.support_level == SupportLevel.UNSUPPORTED
        assert exec_cap.available is False
        assert "Node.js" in exec_cap.missing_adapter or "docker_daemon" in exec_cap.missing_services

        # Discovered tests verification
        evaluator = BaselineEvaluator(str(JS_TS_FIXTURE))
        discovered = evaluator.discover_all_tests()
        assert len(discovered) >= 3
        assert any("multiplies positive integers correctly" in t for t in discovered)
        assert any("calculates exponential power" in t for t in discovered)

    def test_monorepo_workspace_detection(self):
        """Verifies multi-package and pnpm workspace topology resolution."""
        report = CapabilityDetector.generate_report(MONOREPO_FIXTURE)

        assert report.ecosystem.workspace.is_monorepo is True
        assert report.ecosystem.workspace.workspace_type == "pnpm_workspace"
        package_names = [p["name"] for p in report.ecosystem.workspace.packages]
        assert "core" in package_names
        assert "web" in package_names

        # Nested test discovery across packages
        evaluator = BaselineEvaluator(str(MONOREPO_FIXTURE))
        tests = evaluator.discover_all_tests()
        assert len(tests) >= 2
        assert any("formats USD currency correctly" in t for t in tests)
        assert any("handles custom currency symbol" in t for t in tests)

    def test_unsupported_language_classification(self, tmp_path):
        """Verifies that non-scoped ecosystems (e.g. Go, Java, Rust) are explicitly classified as unsupported."""
        unsupported_dir = tmp_path / "go_mock_project"
        unsupported_dir.mkdir(parents=True)
        (unsupported_dir / "main.go").write_text("package main\nfunc main() {}", encoding="utf-8")
        (unsupported_dir / "calc_test.go").write_text("package main\nfunc TestAdd() {}", encoding="utf-8")

        report = CapabilityDetector.generate_report(unsupported_dir)
        assert report.ecosystem.primary_language == "Go"

        # In final scope, Go is unsupported across analysis, test discovery, generation, and docker execution
        assert report.get_stage(PipelineStage.ANALYSIS).support_level == SupportLevel.UNSUPPORTED
        assert "not supported" in report.get_stage(PipelineStage.ANALYSIS).reason.lower()
        assert report.get_stage(PipelineStage.TEST_DISCOVERY).support_level == SupportLevel.UNSUPPORTED
        assert report.get_stage(PipelineStage.TEST_GENERATION).support_level == SupportLevel.UNSUPPORTED
        assert report.get_stage(PipelineStage.DOCKER_EXECUTION).support_level == SupportLevel.UNSUPPORTED
        assert report.get_stage(PipelineStage.EVALUATION).support_level == SupportLevel.UNSUPPORTED

        # Ensure BaselineEvaluator discovers zero tests for unsupported extensions
        evaluator = BaselineEvaluator(str(unsupported_dir))
        assert len(evaluator.discover_all_tests()) == 0

    def test_missing_external_services_detection(self):
        """Verifies detection of backing services (PostgreSQL, Redis) and environment files."""
        report = CapabilityDetector.generate_report(SERVICES_FIXTURE)

        service_names = {s.service_name for s in report.ecosystem.external_services}
        assert "postgresql" in service_names
        assert "redis" in service_names
        assert "docker_daemon" in service_names

        # Environment files
        assert ".env.example" in report.ecosystem.env_files

        # Check remediation notes are populated
        pg_req = next(s for s in report.ecosystem.external_services if s.service_name == "postgresql")
        assert "5432" in pg_req.remediation_notes

    def test_malformed_and_missing_repository(self):
        """Verifies error handling when repository path does not exist."""
        non_existent = Path("C:/invalid/path/does_not_exist_testpilot_99999")
        report = CapabilityDetector.generate_report(non_existent)

        assert report.overall_support_level == SupportLevel.UNSUPPORTED
        assert report.can_run_pipeline is False
        assert report.get_stage(PipelineStage.ONBOARDING).available is False
        assert len(report.blocking_reasons) > 0

    def test_unsupported_test_generation_rejection(self, tmp_path):
        """Verifies that attempting test generation on non-Python repositories fails with clear explanation."""
        mgr = CustomRepositoryBenchmarkManager(storage_dir=str(tmp_path / "store"), workspace_dir=str(tmp_path / "clones"))
        req = RegisterRepositoryRequest(path=str(JS_TS_FIXTURE), name="ts-test-fixture")
        record = mgr.register_repository(req)

        # Record should be partially supported
        assert record.supported_language is False
        assert record.status == RepositoryStatus.PARTIALLY_SUPPORTED
        assert record.stage_capabilities.get("test_generation") == "unsupported"

        # Attempting start_generation_job must raise ValueError with clear missing adapter guidance
        with pytest.raises(ValueError) as excinfo:
            mgr.start_generation_job(record.repo_id)

        assert "Autonomous test generation" in str(excinfo.value)
        assert "unsupported" in str(excinfo.value).lower()

    def test_finmath_compatibility_integrity(self):
        """Verifies that FinMath benchmark suite maintains full Python pytest compatibility."""
        candidate_paths = [
            Path("testbed/evaluation_store/clones/shreshth-mod_finmath").resolve(),
            Path("testbed/repositories/FinMath").resolve(),
        ]
        finmath_path = next((p for p in candidate_paths if p.exists()), None)
        if not finmath_path:
            pytest.skip("FinMath testbed repository not present on disk.")

        report = CapabilityDetector.generate_report(finmath_path)
        assert report.ecosystem.primary_language == "Python"
        assert "pytest" in report.ecosystem.detected_test_frameworks
        assert report.get_stage(PipelineStage.ONBOARDING).support_level == SupportLevel.FULLY_SUPPORTED
        assert report.get_stage(PipelineStage.ANALYSIS).support_level == SupportLevel.FULLY_SUPPORTED
        assert report.get_stage(PipelineStage.TEST_DISCOVERY).support_level == SupportLevel.FULLY_SUPPORTED
        assert report.get_stage(PipelineStage.TEST_GENERATION).support_level == SupportLevel.FULLY_SUPPORTED
        assert report.get_stage(PipelineStage.EVALUATION).support_level == SupportLevel.FULLY_SUPPORTED
