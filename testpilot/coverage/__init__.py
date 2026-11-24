"""
Test Coverage Intelligence subsystem for TestPilot AI.
Provides runtime line coverage, AST-grounded function coverage,
test-to-code mapping, revision comparison, and AI coverage suggestions.
"""

from testpilot.coverage.collector import CoverageCollector
from testpilot.coverage.diff_coverage import RevisionCoverageComparator
from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
from testpilot.coverage.suggestions import CoverageSuggestionEngine
from testpilot.coverage.test_mapping import TestToCodeMapper
from testpilot.coverage.models import (
    CoverageCollectionConfig,
    CoverageEvidenceType,
    CoverageRegression,
    CoverageReport,
    CoverageStatus,
    CoverageTestScenario,
    FileCoverageSummary,
    FileDiffCoverage,
    FunctionCoverage,
    FunctionCoverageSuggestion,
    FunctionStatus,
    LineCoverageData,
    RepositoryCoverageSummary,
    RevisionComparisonReport,
    SuggestionVerificationResult,
    TestToCodeMapping,
)
from testpilot.coverage.storage import CoverageStore

__all__ = [
    "CoverageCollector",
    "FunctionCoverageAnalyzer",
    "TestToCodeMapper",
    "RevisionCoverageComparator",
    "CoverageSuggestionEngine",
    "CoverageStore",
    "CoverageReport",
    "CoverageStatus",
    "CoverageEvidenceType",
    "LineCoverageData",
    "FunctionCoverage",
    "FunctionStatus",
    "FileCoverageSummary",
    "RepositoryCoverageSummary",
    "TestToCodeMapping",
    "CoverageCollectionConfig",
    "CoverageRegression",
    "FileDiffCoverage",
    "RevisionComparisonReport",
    "CoverageTestScenario",
    "FunctionCoverageSuggestion",
    "SuggestionVerificationResult",
]
