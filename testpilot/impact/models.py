"""
Domain models and schemas for Phase B: Multi-Layer Impact Analysis Subsystem.
Defines Change Understanding, Dependency Graph Nodes, Spec & Boundary Impacts,
Categorical Risk Tiers, and Verifiable Impact Evidence Graphs.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    CRITICAL = "CRITICAL"   # Direct call / public API contract / modified boundary
    HIGH = "HIGH"           # 1-hop caller / parameter change / schema mutation
    MEDIUM = "MEDIUM"       # Transitive 2+ hops / internal helper
    LOW = "LOW"             # Structural reachability without verified behavioral coupling
    UNCERTAIN = "UNCERTAIN" # Dynamic dispatch / unannotated fixture / ambiguous caller


class EvidenceType(str, Enum):
    DIRECT_AST_CALL = "DIRECT_AST_CALL"
    TRANSITIVE_CALL_CHAIN = "TRANSITIVE_CALL_CHAIN"
    IMPORT_DEPENDENCY = "IMPORT_DEPENDENCY"
    OPENAPI_ENDPOINT_BINDING = "OPENAPI_ENDPOINT_BINDING"
    AST_BRANCH_BOUNDARY = "AST_BRANCH_BOUNDARY"
    DYNAMIC_FIXTURE_INJECTION = "DYNAMIC_FIXTURE_INJECTION"
    EVENT_DISPATCH = "EVENT_DISPATCH"


class BoundaryImpact(BaseModel):
    """Represents a formal OpenAPI or AST-derived boundary constraint affected by the change."""
    boundary_id: str = Field(default_factory=lambda: f"bnd-{uuid.uuid4().hex[:6]}")
    target_symbol: str
    target_file: str
    boundary_type: str  # e.g. "MIN_VALUE", "MAX_VALUE", "ENUM_VALUE", "NULL_CHECK", "TYPE_VALIDATION"
    parameter_name: Optional[str] = None
    constraint_expression: str
    source_spec: str = "AST_SYNTAX"  # "OPENAPI" or "AST_SYNTAX"
    is_violation_risk: bool = True
    evidence_line: Optional[int] = None


class ImpactGraphNode(BaseModel):
    """A node in the verifiable impact evidence graph."""
    node_id: str
    label: str
    node_type: str  # "CHANGED_SYMBOL", "CALLER_FUNCTION", "BOUNDARY_RULE", "TEST_CASE"
    file_path: str
    line_number: Optional[int] = None
    risk_category: RiskCategory = RiskCategory.MEDIUM
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImpactGraphEdge(BaseModel):
    """A directed edge in the impact evidence graph with explicit provenance."""
    edge_id: str = Field(default_factory=lambda: f"edge-{uuid.uuid4().hex[:6]}")
    source_id: str
    target_id: str
    relationship: str  # "INVOKES", "CONSTRAINS", "TESTS", "DEFINES"
    evidence_type: EvidenceType = EvidenceType.DIRECT_AST_CALL
    call_chain: list[str] = Field(default_factory=list)
    confidence_rationale: str = ""
    is_ambiguous: bool = False


class ImpactEvidenceGraph(BaseModel):
    """Verifiable graph representation of all change impacts and test linkages."""
    graph_id: str = Field(default_factory=lambda: f"graph-{uuid.uuid4().hex[:6]}")
    nodes: list[ImpactGraphNode] = Field(default_factory=list)
    edges: list[ImpactGraphEdge] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0


class ImpactAnalysisReport(BaseModel):
    """
    Authoritative Stage 1 Impact Analysis Report.
    Directly ingested by Stage 2 (Intelligent Test Selection).
    """
    impact_run_id: str = Field(default_factory=lambda: f"imp-{int(time.time())}-{uuid.uuid4().hex[:6]}")
    workflow_id: str
    evolution_run_id: str
    repo_id: str
    repo_path: str
    base_commit_hash: str
    target_commit_hash: str

    # Change Understanding
    changed_files: list[str] = Field(default_factory=list)
    changed_symbols: list[dict[str, Any]] = Field(default_factory=list)

    # Multi-Layer Call Graph & Transitive Blast Radius
    direct_callers: list[dict[str, Any]] = Field(default_factory=list)
    transitive_callers: list[dict[str, Any]] = Field(default_factory=list)
    max_traversal_depth: int = 3

    # Spec & Boundary Impact
    spec_available: bool = False
    spec_path: Optional[str] = None
    boundary_impacts: list[BoundaryImpact] = Field(default_factory=list)

    # Test Impact Mapping
    impacted_test_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Categorical Risk Summary
    risk_distribution: dict[str, int] = Field(
        default_factory=lambda: {
            RiskCategory.CRITICAL.value: 0,
            RiskCategory.HIGH.value: 0,
            RiskCategory.MEDIUM.value: 0,
            RiskCategory.LOW.value: 0,
            RiskCategory.UNCERTAIN.value: 0,
        }
    )

    # Verifiable Evidence Graph
    evidence_graph: ImpactEvidenceGraph = Field(default_factory=ImpactEvidenceGraph)

    # Stage Lifecycle & Timing
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    analysis_duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    critical_warnings: list[str] = Field(default_factory=list)
