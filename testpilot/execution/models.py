"""
Domain models and schemas for Phase D: Isolated Test Execution with Docker (Stage 3).
Defines execution results, disposable container security profiles, and fail-safe Docker diagnostics.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class TestOutcome(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class DockerSecurityProfile(BaseModel):
    """Enforced security controls for isolated container execution."""
    disposable_container: bool = True
    readonly_source_mount: bool = True
    ephemeral_tmp_mount: bool = True
    non_root_user: bool = True
    zero_host_credentials_exposed: bool = True
    network_disabled: bool = True
    cpu_limit: str = "2.0"
    memory_limit: str = "2g"


class DockerDiagnostics(BaseModel):
    """Honest, fail-safe diagnostics when Docker daemon is stopped or unavailable."""
    is_available: bool = False
    daemon_version: Optional[str] = None
    container_engine: str = "Docker Desktop / WSL2"
    error_message: Optional[str] = None
    setup_guidance: list[str] = Field(
        default_factory=lambda: [
            "Start Docker Desktop on Windows host or launch the Docker daemon in WSL2 ('sudo service docker start').",
            "Verify daemon connectivity by running 'docker info' in your terminal.",
            "Once running, retry containerized test execution.",
            "TestPilot refuses to silently execute uncontained code on the host environment for security protection.",
        ]
    )


class TestCaseResult(BaseModel):
    """Execution telemetry for a single test case."""
    test_id: str = Field(default_factory=lambda: f"res-{uuid.uuid4().hex[:6]}")
    test_name: str
    test_file: str
    outcome: TestOutcome = TestOutcome.PASSED
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    targeted_symbol: Optional[str] = None
    exit_code: int = 0


class IsolatedExecutionReport(BaseModel):
    """
    Authoritative Stage 3 Report containing execution telemetry,
    container security profile, and test results passed to Stage 4 Failure Arbitration.
    """
    execution_run_id: str = Field(
        default_factory=lambda: f"exec-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    )
    workflow_id: str
    selection_run_id: str
    repo_id: str
    repo_path: str
    base_commit_hash: str
    target_commit_hash: str

    # Execution Counts
    total_executed: int = 0
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0

    # Test Results
    results: list[TestCaseResult] = Field(default_factory=list)
    failed_test_names: list[str] = Field(default_factory=list)

    # Security & Diagnostics
    is_docker_isolated: bool = True
    security_profile: DockerSecurityProfile = Field(default_factory=DockerSecurityProfile)
    docker_diagnostics: DockerDiagnostics = Field(default_factory=DockerDiagnostics)

    # Lifecycle & Timing
    execution_duration_ms: float = 0.0
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    warnings: list[str] = Field(default_factory=list)
