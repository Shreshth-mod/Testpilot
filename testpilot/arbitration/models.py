"""
Domain models and schemas for Phase E: Failure Arbitration Subsystem (Stage 4).
Defines the 6 evidence-based failure categories, base-vs-target comparative telemetry,
and developer review confirmation workflows.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class ArbitrationCategory(str, Enum):
    REGRESSION = "REGRESSION"         # Failure introduced by change under test
    PRE_EXISTING = "PRE_EXISTING"     # Already failing on base commit
    FLAKY = "FLAKY"                   # Non-deterministic / timing / concurrency
    INFRASTRUCTURE = "INFRASTRUCTURE" # Environment, timeout, missing dependency, sandbox limitation
    TEST_DEFECT = "TEST_DEFECT"       # Faulty test assertion / broken test harness
    UNCERTAIN = "UNCERTAIN"           # Insufficient evidence for conclusive classification


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    DISPUTED = "DISPUTED"
    OVERRIDDEN = "OVERRIDDEN"


class ArbitrationItem(BaseModel):
    """Evidence-grounded triage record for a single failed test."""
    item_id: str = Field(default_factory=lambda: f"arb-{uuid.uuid4().hex[:6]}")
    test_name: str
    test_file: str
    category: ArbitrationCategory = ArbitrationCategory.UNCERTAIN
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    target_outcome: str = "FAILED"
    base_outcome: Optional[str] = None  # "PASSED" -> REGRESSION, "FAILED" -> PRE_EXISTING
    failure_signature: str = ""
    root_cause_explanation: str = ""
    recommended_action: str = ""
    evidence: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: Optional[str] = None
    overridden_category: Optional[ArbitrationCategory] = None


class FailureArbitrationReport(BaseModel):
    """
    Authoritative Stage 4 Report triaging isolated container test failures into
    6 actionable categories with base-vs-target provenance.
    """
    arbitration_run_id: str = Field(
        default_factory=lambda: f"arb-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    )
    workflow_id: str
    execution_run_id: str
    repo_id: str
    repo_path: str
    base_commit_hash: str
    target_commit_hash: str

    total_failures_triaged: int = 0
    category_counts: dict[str, int] = Field(
        default_factory=lambda: {
            ArbitrationCategory.REGRESSION.value: 0,
            ArbitrationCategory.PRE_EXISTING.value: 0,
            ArbitrationCategory.FLAKY.value: 0,
            ArbitrationCategory.INFRASTRUCTURE.value: 0,
            ArbitrationCategory.TEST_DEFECT.value: 0,
            ArbitrationCategory.UNCERTAIN.value: 0,
        }
    )
    items: list[ArbitrationItem] = Field(default_factory=list)

    # Lifecycle & Timing
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    analysis_duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
