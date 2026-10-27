"""
Models for TestPilot Cross-Repository Capabilities and Pipeline Stage Classification.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SupportLevel(str, Enum):
    """Granular support classification for a repository or pipeline stage."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class PipelineStage(str, Enum):
    """The discrete operational stages of TestPilot's repository intelligence pipeline."""

    ONBOARDING = "onboarding"
    ANALYSIS = "analysis"
    TEST_DISCOVERY = "test_discovery"
    TEST_GENERATION = "test_generation"
    GROUND_TRUTH_VALIDATION = "ground_truth_validation"
    DOCKER_EXECUTION = "docker_execution"
    EVALUATION = "evaluation"


class EcosystemType(str, Enum):
    """Software ecosystems recognized by TestPilot (Python and JavaScript/TypeScript only)."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class StageCapability(BaseModel):
    """
    Detailed capability assessment for a single pipeline stage on a given repository.
    Includes precise reasons, missing adapters, missing external services, and remediation guidance.
    """

    stage: PipelineStage
    support_level: SupportLevel
    available: bool
    reason: str
    missing_adapter: Optional[str] = None
    missing_services: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class ManifestInfo(BaseModel):
    """Metadata describing a detected project configuration or dependency manifest."""

    manifest_type: str  # e.g., 'pyproject.toml', 'package.json', 'go.mod', 'pom.xml'
    path: str           # Relative path from repository root
    ecosystem: str      # e.g., 'python', 'javascript', 'typescript', 'go', 'java', 'rust'
    package_manager: str  # e.g., 'pip', 'poetry', 'npm', 'pnpm', 'yarn', 'go modules', 'maven', 'cargo'
    dependencies: list[str] = Field(default_factory=list)
    dev_dependencies: list[str] = Field(default_factory=list)
    test_runner_command: Optional[str] = None


class WorkspaceStructure(BaseModel):
    """Workspace or monorepo topology of the target repository."""

    is_monorepo: bool = False
    workspace_type: Optional[str] = None  # e.g. 'npm_workspaces', 'pnpm_workspace', 'cargo_workspace', 'go_workspaces', 'single_package'
    root_manifest: Optional[str] = None
    packages: list[dict[str, str]] = Field(default_factory=list)  # list of {"name": ..., "path": ..., "manifest": ...}


class ExternalServiceRequirement(BaseModel):
    """An external backing service or infrastructure component required by the repository."""

    service_name: str  # e.g., 'postgresql', 'redis', 'docker_daemon', 'mongodb', 'mysql'
    detected_via: str  # e.g., 'docker-compose.yml', 'DATABASE_URL in .env.example', 'redis dependency in pyproject.toml'
    is_available: bool = False
    is_critical: bool = False
    remediation_notes: str = ""


class RepositoryEcosystem(BaseModel):
    """Aggregated ecosystem intelligence for an inspected codebase."""

    primary_language: str = "Unknown"
    detected_languages: list[str] = Field(default_factory=list)
    manifests: list[ManifestInfo] = Field(default_factory=list)
    workspace: WorkspaceStructure = Field(default_factory=WorkspaceStructure)
    detected_test_frameworks: list[str] = Field(default_factory=list)
    external_services: list[ExternalServiceRequirement] = Field(default_factory=list)
    env_files: list[str] = Field(default_factory=list)
    ci_configs: list[str] = Field(default_factory=list)


class RepositoryCompatibilityReport(BaseModel):
    """
    Authoritative cross-repository compatibility audit report.
    Presents per-stage support levels, explicit failure causes, and actionable recommendations.
    """

    repo_name: str
    canonical_path: str
    ecosystem: RepositoryEcosystem
    stage_capabilities: dict[str, StageCapability] = Field(default_factory=dict)
    overall_support_level: SupportLevel
    can_run_pipeline: bool
    summary: str
    unsupported_stages: list[str] = Field(default_factory=list)
    partially_supported_stages: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    def get_stage(self, stage: PipelineStage | str) -> Optional[StageCapability]:
        stage_key = stage.value if isinstance(stage, PipelineStage) else str(stage)
        return self.stage_capabilities.get(stage_key)

    def is_stage_supported(self, stage: PipelineStage | str) -> bool:
        cap = self.get_stage(stage)
        return cap is not None and cap.available and cap.support_level == SupportLevel.FULLY_SUPPORTED
