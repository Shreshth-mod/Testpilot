"""
Domain models and schemas for Phase C: Intelligent Test Selection & Targeted Generation.
Provides models for ranking existing test suites, calculating selection efficiency,
and validating synthesized targeted tests against the 5-criterion standard.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class FileChangeClassification(str, Enum):
    """Categorical classification of modified or affected source files."""
    BUSINESS_LOGIC = "business_logic"
    API_INTERFACE = "api_interface"
    VALIDATION_BOUNDARY = "validation_boundary"
    ERROR_HANDLING = "error_handling"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    TESTS_ONLY = "tests_only"
    DOCUMENTATION_ONLY = "documentation_only"
    OTHER = "other"


class AffectedFileAnalysis(BaseModel):
    """Comprehensive analysis record for an affected file, its classification, and verification need."""
    file_path: str
    change_type: FileChangeClassification = FileChangeClassification.BUSINESS_LOGIC
    reason_affected: str
    behavior_requiring_verification: str
    is_uncovered: bool = False
    coverage_status: str = "needs_generated_tests"  # "covered_by_existing_tests", "needs_generated_tests", "non_behavioral_no_test_needed"
    rationale: str = ""
    target_symbols: list[str] = Field(default_factory=list)


class SelectedTest(BaseModel):
    """An existing regression test selected based on Stage 1 impact evidence."""
    test_id: str = Field(default_factory=lambda: f"test-{uuid.uuid4().hex[:6]}")
    test_name: str
    test_file: str
    line_number: Optional[int] = None
    targeted_symbol: str
    selection_tier: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    selection_reason: str
    selection_criteria: list[str] = Field(default_factory=list)
    execution_command: str
    estimated_duration_ms: float = 50.0
    is_direct: bool = True


class TargetedTest(BaseModel):
    """
    A newly synthesized targeted unit test covering an unexercised impacted behavior.
    Enforces the 5-criterion qualification standard with strict isolation from evaluation benchmarks.
    Never labeled as passed until actual execution succeeds.
    """
    test_id: str = Field(default_factory=lambda: f"tgt-{uuid.uuid4().hex[:6]}")
    test_name: str
    target_file: str = ""
    target_symbol: str
    target_boundary: Optional[str] = None
    uncovered_behavior: str = ""
    rationale: str = ""
    test_category: str = "contract_correctness"  # contract_correctness, input_output, branch_condition, boundary_value, error_handling, caller_compatibility, historical_regression
    code: str
    file_path: str
    qualification_passed: bool = False
    qualification_details: dict[str, bool] = Field(
        default_factory=lambda: {
            "ast_syntax_valid": False,
            "collectable": False,
            "has_non_trivial_assertions": False,
            "contract_grounded": False,
            "isolated_from_benchmark": True,
        }
    )
    failure_reasons: list[str] = Field(default_factory=list)
    execution_status: str = "NOT_RUN"  # "NOT_RUN", "QUEUED", "RUNNING", "PASSED", "FAILED", "ERROR", "BLOCKED"
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None


class SelectionEfficiency(BaseModel):
    """Metrics demonstrating test suite reduction and CI runtime savings."""
    total_repository_tests: int = 0
    selected_existing_tests: int = 0
    suite_reduction_percentage: float = 0.0
    estimated_full_suite_duration_s: float = 0.0
    estimated_selected_duration_s: float = 0.0
    time_savings_percentage: float = 0.0


class TestSelectionReport(BaseModel):
    """
    Authoritative Stage 2 Report containing affected files analysis, selected existing tests,
    targeted generated tests, and suite reduction metrics.
    """
    selection_run_id: str = Field(
        default_factory=lambda: f"sel-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    )
    workflow_id: str
    impact_run_id: str
    repo_id: str
    repo_path: str
    base_commit_hash: str
    target_commit_hash: str

    # Affected Files Analysis
    affected_files: list[AffectedFileAnalysis] = Field(default_factory=list)
    affected_files_count: int = 0

    # Selection Results
    selected_tests: list[SelectedTest] = Field(default_factory=list)
    targeted_generated_tests: list[TargetedTest] = Field(default_factory=list)
    uncovered_impacted_symbols: list[str] = Field(default_factory=list)

    # Detailed Count Breakdown
    existing_tests_selected_count: int = 0
    new_tests_generated_count: int = 0
    tests_qualified_count: int = 0
    tests_executed_count: int = 0
    tests_passed_count: int = 0
    tests_failed_count: int = 0

    # Efficiency Metrics
    efficiency: SelectionEfficiency = Field(default_factory=SelectionEfficiency)

    # Execution Command Set (ready for Stage 3 Isolated Container Execution)
    aggregated_test_command: str = ""

    # Lifecycle & Timing
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    analysis_duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
