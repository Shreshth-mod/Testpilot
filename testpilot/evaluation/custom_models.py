"""
Data models for the Dynamic Multi-Repository Regression Benchmark Manager.
Provides schemas for custom repository registration, structure inspection,
independent benchmark ground truth definitions, and per-repository evaluation records.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from testpilot.evaluation.models import EvaluationRun, GroundTruth


class RepositoryStatus(str, Enum):
    """Lifecycle and benchmark-readiness states for a registered custom repository."""

    REGISTERED = "registered"
    INSPECTING = "inspecting"
    ANALYZED = "analyzed"
    TESTS_DISCOVERED = "tests_discovered"
    GENERATING_TESTS = "generating_tests"
    VALIDATING_TESTS = "validating_tests"
    BENCHMARK_PREPARATION_REQUIRED = "benchmark_preparation_required"
    BENCHMARK_UNDER_VALIDATION = "benchmark_under_validation"
    READY_FOR_EVALUATION = "ready_for_evaluation"
    EVALUATION_RUNNING = "evaluation_running"
    EVALUATED = "evaluated"
    EVALUATION_FAILED = "evaluation_failed"
    PARTIALLY_SUPPORTED = "partially_supported"
    FAILED = "failed"
    STALE = "stale"


class BenchmarkSourceType(str, Enum):
    """Independent origin and provenance classification for a benchmark case."""

    HISTORICAL_BUG_FIX = "historical_bug_fix"
    GIT_COMMIT_RANGE = "git_commit_range"
    CURATED_FIXTURE = "curated_fixture"
    MANUAL_MAPPING = "manual_mapping"
    MUTATION_ANALYSIS = "mutation_analysis"
    BEHAVIORAL_CONTRACT = "behavioral_contract"


class GroundTruthValidationLevel(str, Enum):
    """Validation level for independent benchmark ground truth."""

    HISTORICAL_REGRESSION_VERIFIED = "historical_regression_verified"
    CONTRACT_MUTATION_VERIFIED = "contract_mutation_verified"
    PROPOSED_AWAITING_REVIEW = "proposed_awaiting_review"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GroundTruthReviewStatus(str, Enum):
    """Human and automated review lifecycle status for benchmark ground truth."""

    CANDIDATE = "candidate"
    EVIDENCE_COLLECTION = "evidence_collection"
    UNDER_VALIDATION = "under_validation"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    STALE = "stale"


class GeneratedTestRecord(BaseModel):
    """Provenance, target mapping, and validation details for a generated test."""

    test_id: str
    target_symbol: str
    target_file: str
    test_file: str
    test_name: str
    intended_behavior: str
    generation_method: str = "ast_contract_analysis"
    source_evidence: str = ""
    syntax_valid: bool = True
    executes_successfully: bool = False
    is_trivial: bool = False
    mutation_sensitive: bool = False
    rejection_reason: Optional[str] = None
    created_at: str = ""


class CustomBenchmarkCase(BaseModel):
    """
    An independent, versioned benchmark ground-truth case for a custom repository.
    Independent of predictions: must contain traceable provenance and verified expected callers.
    """

    case_id: str
    name: str
    description: str = ""
    source_type: BenchmarkSourceType = BenchmarkSourceType.GIT_COMMIT_RANGE
    base_commit: str = "HEAD~1"
    target_commit: str = "HEAD"
    commit_sha: Optional[str] = None
    ground_truth: GroundTruth
    validation_level: GroundTruthValidationLevel = GroundTruthValidationLevel.CONTRACT_MUTATION_VERIFIED
    review_status: GroundTruthReviewStatus = GroundTruthReviewStatus.VALIDATED
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    validation_evidence: dict[str, Any] = Field(default_factory=dict)
    mutant_kill_rate: float = 1.0
    fault_detection_rate: Optional[float] = 1.0
    limitations_notes: Optional[str] = None
    total_tests_suite: Optional[int] = None
    created_at: str = ""


class CustomRepositoryRecord(BaseModel):
    """
    Persistent record representing a registered custom repository in the benchmark suite.
    Tracks canonical identity, git revision, test discovery, ground truth cases, and evaluation runs.
    """

    repo_id: str
    name: str
    canonical_path: str
    remote_url: Optional[str] = None
    is_remote: bool = False
    is_git: bool = True
    detected_revision: Optional[str] = None
    current_branch: Optional[str] = None
    is_active: bool = True
    status: RepositoryStatus = RepositoryStatus.REGISTERED
    status_message: str = "Repository registered; pending inspection and analysis."
    languages: list[str] = Field(default_factory=list)
    source_files_count: int = 0
    test_files_count: int = 0
    discovered_tests: list[str] = Field(default_factory=list)
    total_tests_count: int = 0
    test_framework: str = "pytest"
    baseline_test_status: str = "untested"
    baseline_test_error: Optional[str] = None
    supported_language: bool = True
    unsupported_reason: Optional[str] = None
    generated_tests: list[GeneratedTestRecord] = Field(default_factory=list)
    generated_tests_count: int = 0
    validated_tests_count: int = 0
    mutation_sensitive_tests_count: int = 0
    cases_awaiting_review_count: int = 0
    benchmark_cases: list[CustomBenchmarkCase] = Field(default_factory=list)
    validated_benchmark_cases_count: int = 0
    last_analyzed_at: Optional[str] = None
    last_evaluated_at: Optional[str] = None
    latest_evaluation_runs: dict[str, EvaluationRun] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class RegisterRepositoryRequest(BaseModel):
    """Payload to register a new repository in the benchmark suite (supports local path or remote GitHub URL)."""

    path: str = Field(..., description="Local directory path or remote GitHub repository URL")
    name: Optional[str] = None


class PrepareBenchmarkRequest(BaseModel):
    """Payload to prepare or validate an independent ground truth case."""

    name: str
    description: str = ""
    source_type: BenchmarkSourceType = BenchmarkSourceType.GIT_COMMIT_RANGE
    base_commit: str = "HEAD~1"
    target_commit: str = "HEAD"
    expected_changed_symbols: list[str] = Field(default_factory=list)
    expected_impacted_symbols: list[str] = Field(default_factory=list)
    expected_test_callers: list[str] = Field(default_factory=list)
    provenance: str
    notes: str = ""


class ReviewGroundTruthRequest(BaseModel):
    """Payload for human review and approval/rejection of benchmark ground truth."""

    action: str = Field(..., description="'approve' to mark validated, or 'reject' to exclude")
    reviewer: str = Field(default="Human Reviewer", description="Name/ID of reviewer")
    notes: Optional[str] = Field(default=None, description="Review notes or justification")


class PrepareHistoricalBugRequest(BaseModel):
    """Payload to build Method A historical bug benchmark ground truth."""

    name: str
    description: str = ""
    buggy_commit: str = Field(..., description="Commit SHA of pre-fix buggy state")
    corrected_commit: str = Field(..., description="Commit SHA of post-fix corrected state")
    test_id: str = Field(..., description="Identifier of the regression test distinguishing behavior")
    expected_changed_symbols: list[str] = Field(default_factory=list)
    expected_impacted_symbols: list[str] = Field(default_factory=list)
    provenance: str = "Historical bug fix verification via git commit comparison"
    notes: str = ""
