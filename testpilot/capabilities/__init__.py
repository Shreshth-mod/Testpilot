"""
TestPilot Capabilities & Cross-Repository Ecosystem Intelligence.
Provides manifest detection, monorepo workspace resolution, external dependency scanning,
and granular per-stage pipeline capability classification.
"""

from testpilot.capabilities.models import (
    EcosystemType,
    ExternalServiceRequirement,
    ManifestInfo,
    PipelineStage,
    RepositoryCompatibilityReport,
    RepositoryEcosystem,
    StageCapability,
    SupportLevel,
    WorkspaceStructure,
)
from testpilot.capabilities.detector import CapabilityDetector

__all__ = [
    "EcosystemType",
    "ExternalServiceRequirement",
    "ManifestInfo",
    "PipelineStage",
    "RepositoryCompatibilityReport",
    "RepositoryEcosystem",
    "StageCapability",
    "SupportLevel",
    "WorkspaceStructure",
    "CapabilityDetector",
]
