"""
Domain models and schemas for TestPilot Test Coverage Intelligence.
Defines line-level, function-level, and test-mapping coverage telemetry,
storage records, and multi-granularity coverage reports.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class CoverageStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    UNSUPPORTED_MODE = "UNSUPPORTED_MODE"


class FunctionStatus(str, Enum):
    FULLY_COVERED = "FULLY_COVERED"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    UNCOVERED = "UNCOVERED"


class CoverageEvidenceType(str, Enum):
    RUNTIME_COVERAGE = "RUNTIME_COVERAGE"      # Empirically measured line-level execution
    DIRECT_AST_CALL = "DIRECT_AST_CALL"        # Static AST direct call-site reference
    TRANSITIVE_CHAIN = "TRANSITIVE_CHAIN"      # Multi-hop static AST reachability
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"  # Vector embedding / RAG suggestion


class LineCoverageData(BaseModel):
    """Line-level coverage telemetry for a single source file."""
    file_path: str                             # Normalized relative repository path
    absolute_path: str = ""
    executed_lines: list[int] = Field(default_factory=list)
    missing_lines: list[int] = Field(default_factory=list)
    excluded_lines: list[int] = Field(default_factory=list)
    executable_lines: list[int] = Field(default_factory=list)
    total_statements: int = 0
    covered_statements: int = 0
    line_coverage_pct: float = 0.0


class FunctionCoverage(BaseModel):
    """Function and method level coverage analysis derived from AST boundaries and line coverage."""
    function_name: str
    qualified_name: str                        # e.g., "OrderService.calculate_total" or "helper_fn"
    file_path: str
    class_name: Optional[str] = None
    start_line: int
    end_line: int
    executable_lines: list[int] = Field(default_factory=list)
    executed_lines: list[int] = Field(default_factory=list)
    missing_lines: list[int] = Field(default_factory=list)
    coverage_pct: float = 0.0
    status: FunctionStatus = FunctionStatus.UNCOVERED
    is_nested: bool = False
    executing_tests: list[str] = Field(default_factory=list)


class FileCoverageSummary(BaseModel):
    """File-level aggregation combining line and function coverage."""
    file_path: str
    total_statements: int = 0
    covered_statements: int = 0
    missing_statements: int = 0
    line_coverage_pct: float = 0.0
    total_functions: int = 0
    covered_functions: int = 0
    partially_covered_functions: int = 0
    uncovered_functions: int = 0
    function_coverage_pct: float = 0.0
    functions: list[FunctionCoverage] = Field(default_factory=list)


class RepositoryCoverageSummary(BaseModel):
    """Repository-wide coverage aggregation."""
    repo_id: str
    commit_hash: str
    workflow_id: Optional[str] = None
    total_files: int = 0
    total_statements: int = 0
    covered_statements: int = 0
    missing_statements: int = 0
    overall_line_coverage_pct: float = 0.0
    total_functions: int = 0
    covered_functions: int = 0
    partially_covered_functions: int = 0
    uncovered_functions: int = 0
    overall_function_coverage_pct: float = 0.0


class TestToCodeMapping(BaseModel):
    """Direct mapping from a test execution to the source code it touches."""
    test_id: str                               # e.g. "tests/test_order.py::test_shipping"
    test_name: str
    test_file: str
    covered_files: list[str] = Field(default_factory=list)
    covered_functions: list[str] = Field(default_factory=list)
    executed_lines_by_file: dict[str, list[int]] = Field(default_factory=dict)
    evidence_type: CoverageEvidenceType = CoverageEvidenceType.RUNTIME_COVERAGE


class CoverageCollectionConfig(BaseModel):
    """Configuration for a coverage collection run."""
    source_paths: Optional[list[str]] = None
    omit_patterns: list[str] = Field(
        default_factory=lambda: [
            "*/tests/*",
            "*test_*.py",
            "*_test.py",
            "*/.venv/*",
            "*/venv/*",
            "*/__pycache__/*",
            "*/site-packages/*",
            "conftest.py",
            "setup.py",
        ]
    )
    branch_coverage: bool = False
    timeout_seconds: int = 60


class CoverageReport(BaseModel):
    """
    Authoritative Test Coverage Intelligence Report.
    Persisted to database and JSON cache, scoped by repository, revision, and workflow.
    """
    coverage_run_id: str = Field(
        default_factory=lambda: f"cov-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    )
    workflow_id: Optional[str] = None
    execution_run_id: Optional[str] = None
    repo_id: str = "default"
    repo_path: str = "."
    commit_hash: str = "HEAD"
    execution_mode: str = "local_subprocess"
    is_full_suite: bool = False
    status: CoverageStatus = CoverageStatus.SUCCESS
    error_message: Optional[str] = None
    test_cases: list[str] = Field(default_factory=list)

    # Line Level Aggregation
    total_statements: int = 0
    covered_statements: int = 0
    missing_statements: int = 0
    line_coverage_pct: float = 0.0

    # Function Level Aggregation
    total_functions: int = 0
    covered_functions: int = 0
    partially_covered_functions: int = 0
    uncovered_functions: int = 0
    function_coverage_pct: float = 0.0

    # Detailed per-file data
    files: dict[str, LineCoverageData] = Field(default_factory=dict)
    file_summaries: list[FileCoverageSummary] = Field(default_factory=list)
    test_mappings: list[TestToCodeMapping] = Field(default_factory=list)

    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageRegression(BaseModel):
    """Details a function that lost coverage between revisions."""
    file_path: str
    function_name: str
    previous_coverage_pct: float
    current_coverage_pct: float
    previous_status: FunctionStatus
    current_status: FunctionStatus
    lines_lost: list[int] = Field(default_factory=list)


class FileDiffCoverage(BaseModel):
    """File-level delta between base and target revisions."""
    file_path: str
    change_type: str = "modified"  # added, modified, deleted, renamed, unchanged
    base_line_coverage_pct: Optional[float] = None
    target_line_coverage_pct: Optional[float] = None
    line_coverage_delta: Optional[float] = None
    base_function_coverage_pct: Optional[float] = None
    target_function_coverage_pct: Optional[float] = None
    function_coverage_delta: Optional[float] = None
    newly_covered_functions: list[str] = Field(default_factory=list)
    newly_uncovered_functions: list[str] = Field(default_factory=list)
    uncovered_changed_lines: list[int] = Field(default_factory=list)


class RevisionComparisonReport(BaseModel):
    """Authoritative revision comparison report between two git commits or coverage runs."""
    base_commit: str
    target_commit: str
    base_run_id: Optional[str] = None
    target_run_id: Optional[str] = None
    base_available: bool = True
    line_coverage_delta: float = 0.0
    function_coverage_delta: float = 0.0
    base_line_coverage_pct: float = 0.0
    target_line_coverage_pct: float = 0.0
    base_function_coverage_pct: float = 0.0
    target_function_coverage_pct: float = 0.0
    regressions: list[CoverageRegression] = Field(default_factory=list)
    newly_covered_functions: list[str] = Field(default_factory=list)
    newly_uncovered_functions: list[str] = Field(default_factory=list)
    uncovered_new_or_modified_functions: list[str] = Field(default_factory=list)
    files: list[FileDiffCoverage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class CoverageTestScenario(BaseModel):
    """A concrete test scenario formulated to exercise specific uncovered logic."""
    scenario_id: str = Field(default_factory=lambda: f"scen-{uuid.uuid4().hex[:6]}")
    title: str
    target_function: str
    target_file: str
    rationale: str
    uncovered_lines_targeted: list[int] = Field(default_factory=list)
    suggested_inputs: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str
    sample_test_code: Optional[str] = None


class FunctionCoverageSuggestion(BaseModel):
    """Targeted AI suggestion package for an uncovered or partially covered function."""
    function_name: str
    qualified_name: str
    file_path: str
    current_coverage_pct: float
    current_status: FunctionStatus
    missing_lines: list[int] = Field(default_factory=list)
    scenarios: list[CoverageTestScenario] = Field(default_factory=list)
    synthesized_test_code: Optional[str] = None


class SuggestionVerificationResult(BaseModel):
    """Result of generating and executing suggested tests to empirically verify coverage gain."""
    target_function: str
    target_file: str
    generated_test_file: str
    tests_executed: list[str] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    coverage_before_pct: float = 0.0
    coverage_after_pct: float = 0.0
    coverage_delta_pct: float = 0.0
    coverage_improved: bool = False
    verified: bool = False
    details: str = ""


