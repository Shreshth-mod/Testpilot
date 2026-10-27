"""
TestPilot Evaluation Subsystem.
Provides empirical benchmarking, baseline comparisons (Full Regression,
Naive Name-Based Selection, TestPilot Qualified Identity), deterministic
ground-truth evaluation, and component-level correctness verification.
"""

from testpilot.evaluation.engine import EvaluationEngine
from testpilot.evaluation.models import (
    BaselineResult,
    BaselineType,
    BenchmarkCase,
    ComponentValidationResult,
    EvaluationMetrics,
    EvaluationRun,
    GroundTruth,
    TestSelectionEvidence,
)
from testpilot.evaluation.storage import EvaluationStorage

from testpilot.evaluation.custom_manager import CustomRepositoryBenchmarkManager
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    CustomRepositoryRecord,
    GroundTruthReviewStatus,
    GroundTruthValidationLevel,
    PrepareBenchmarkRequest,
    PrepareHistoricalBugRequest,
    RegisterRepositoryRequest,
    RepositoryStatus,
    ReviewGroundTruthRequest,
)

__all__ = [
    "BaselineResult",
    "BaselineType",
    "BenchmarkCase",
    "BenchmarkSourceType",
    "ComponentValidationResult",
    "CustomBenchmarkCase",
    "CustomRepositoryBenchmarkManager",
    "CustomRepositoryRecord",
    "EvaluationEngine",
    "EvaluationMetrics",
    "EvaluationRun",
    "EvaluationStorage",
    "GroundTruth",
    "GroundTruthReviewStatus",
    "GroundTruthValidationLevel",
    "PrepareBenchmarkRequest",
    "PrepareHistoricalBugRequest",
    "RegisterRepositoryRequest",
    "RepositoryStatus",
    "ReviewGroundTruthRequest",
    "TestSelectionEvidence",
]
