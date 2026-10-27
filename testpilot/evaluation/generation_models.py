"""
Data models for Durable Test Generation Jobs, Repository Validation Summaries,
and Historical Bug-Fix Ground Truth Mining.
"""

from enum import Enum
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    GeneratedTestRecord,
    GroundTruthReviewStatus,
    GroundTruthValidationLevel,
)
from testpilot.evaluation.models import GroundTruth


class GenerationJobStatus(str, Enum):
    """Explicit, persisted states for the test-generation job lifecycle."""

    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    DISCOVERING_TESTS = "DISCOVERING_TESTS"
    GENERATING = "GENERATING"
    QUALIFYING = "QUALIFYING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GenerationJobRecord(BaseModel):
    """
    Durable, observable record tracking a machine test generation job.
    Persisted to disk and restored across browser reloads or server restarts.
    """

    job_id: str
    repo_id: str
    commit_sha: Optional[str] = None
    status: GenerationJobStatus = GenerationJobStatus.QUEUED
    current_stage: str = "Queued for processing"
    started_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    completed_at: Optional[str] = None
    last_progress_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    stage_durations: dict[str, float] = Field(default_factory=dict)
    candidates_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    generated_test_records: list[GeneratedTestRecord] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    diagnostic_messages: list[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None
    workflow_id: Optional[str] = None
    execution_run_id: Optional[str] = None


class RepositoryValidationSummary(BaseModel):
    """
    Pre-registration inspection summary presented to user in guided onboarding.
    Validates repository existence, accessibility, languages, and existing tests before saving.
    """

    source_type: str  # 'public_git', 'private_git', 'local_folder'
    repo_name: str
    remote_url: Optional[str] = None
    canonical_path: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    languages: list[str] = Field(default_factory=list)
    test_framework: str = "pytest"
    existing_tests_count: int = 0
    source_files_count: int = 0
    size_bytes: int = 0
    excluded_dirs: list[str] = Field(default_factory=list)
    is_monorepo: bool = False
    workspace_type: Optional[str] = None
    manifests: list[str] = Field(default_factory=list)
    stage_capabilities: dict[str, str] = Field(default_factory=dict)
    compatibility_report: Optional[dict[str, Any]] = None
    private_access_status: str = "not_required"  # 'not_required', 'authenticated', 'auth_required', 'unsupported'
    is_valid: bool = True
    warnings: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class HistoricalBugCandidate(BaseModel):
    """A mined bug-fix candidate with before/after commit comparison and validation status."""

    candidate_id: str
    repo_id: str
    commit_sha: str
    parent_commit_sha: str
    commit_message: str
    changed_files: list[str] = Field(default_factory=list)
    changed_symbols: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    candidate_tests: list[str] = Field(default_factory=list)
    before_result: Optional[str] = None  # e.g., 'FAILED' (expected), 'PASSED', 'ERROR'
    after_result: Optional[str] = None   # e.g., 'PASSED' (expected), 'FAILED', 'ERROR'
    validation_status: GroundTruthReviewStatus = GroundTruthReviewStatus.CANDIDATE
    validation_level: GroundTruthValidationLevel = GroundTruthValidationLevel.PROPOSED_AWAITING_REVIEW
    confidence: float = 0.5
    evidence_notes: str = ""
