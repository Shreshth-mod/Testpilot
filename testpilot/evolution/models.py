"""
Pydantic v2 schemas and domain models for Repository Evolution Intelligence.
Provides data structures for Git diff change analysis, transitive impact graphs,
evidence trails, uncertainty quantification, and test prioritization.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ChangeType(str, Enum):
    MODIFIED = "MODIFIED"
    ADDED = "ADDED"
    DELETED = "DELETED"


class ImpactType(str, Enum):
    DIRECT = "DIRECT"                           # 1-hop caller / direct reference
    INDIRECT = "INDIRECT"                       # 2+ hops transitive caller
    EVENT_REGISTRATION = "EVENT_REGISTRATION"   # Registered callback / signal receiver


class PriorityTier(str, Enum):
    CRITICAL = "CRITICAL"   # P1: Tests calling modified symbol directly
    HIGH = "HIGH"           # P2: Tests calling 1-hop direct callers
    MEDIUM = "MEDIUM"       # P3: Tests calling indirect callers or sharing schema
    LOW = "LOW"             # P4: Regression tests in same package/domain


class SemanticImpactClassification(str, Enum):
    """
    Evidence-grounded classification of the behavioral relationship between a changed symbol
    and a structurally affected symbol:
    - DIRECT_BEHAVIORAL_DEPENDENCY: Evidence shows the symbol directly relies on the changed behavior or contract.
    - INDIRECT_BEHAVIORAL_DEPENDENCY: Evidence establishes behavioral dependence through intermediate symbols or data/control flow.
    - STRUCTURAL_ONLY: A structural relationship exists, but evidence does not establish meaningful behavioral dependence.
    - UNCERTAIN: Evidence is insufficient, conflicting, incomplete, or unavailable.
    """
    DIRECT_BEHAVIORAL_DEPENDENCY = "DIRECT_BEHAVIORAL_DEPENDENCY"
    INDIRECT_BEHAVIORAL_DEPENDENCY = "INDIRECT_BEHAVIORAL_DEPENDENCY"
    STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
    UNCERTAIN = "UNCERTAIN"


class EvidenceSourceType(str, Enum):
    """Provenance and mechanism of supporting evidence for semantic classification."""
    DIRECTLY_OBSERVED = "DIRECTLY_OBSERVED"     # Directly observed in AST / call-site data-flow
    STATICALLY_INFERRED = "STATICALLY_INFERRED" # Inferred across multi-hop call graph or contracts
    RETRIEVED_SEMANTIC = "RETRIEVED_SEMANTIC"   # Retrieved via repository code vector embeddings (RAG)
    MODEL_INTERPRETED = "MODEL_INTERPRETED"     # Interpreted by CodeLlama / language model


class SemanticEvidenceRef(BaseModel):
    """Specific verifiable evidence citation supporting a semantic impact classification."""
    source_file: str = Field(..., description="Repository-relative file path")
    line_start: Optional[int] = Field(default=None, description="Starting line number")
    line_end: Optional[int] = Field(default=None, description="Ending line number")
    symbol_name: Optional[str] = Field(default=None, description="Associated symbol name")
    evidence_type: str = Field(..., description="E.g., RETURN_VALUE_CONSUMPTION, CONTROL_FLOW_CONDITION, PARAMETER_PASSING, LOGGING_ONLY_CALL, TEST_ASSERTION")
    snippet: str = Field(default="", description="Verifiable code snippet or assertion")
    retrieval_method: str = Field(default="ast_syntax", description="Method: ast_syntax, dataflow_analysis, repo_vector_rag, sourcegraph_ref, or llm_inference")
    is_verified: bool = Field(default=True, description="Whether the evidence reference was verified against actual source/context")


class PrioritizationRank(BaseModel):
    """Transparent lexicographic rank and justification for function importance."""
    rank: int = Field(..., description="1-based prioritization order")
    tier: str = Field(..., description="CRITICAL_BEHAVIORAL, HIGH_BEHAVIORAL, TRANSITIVE_BEHAVIORAL, STRUCTURAL_BASELINE, or UNCERTAIN_INVESTIGATE")
    reason: str = Field(..., description="Transparent, human-readable rationale for ranking position")
    criteria: list[str] = Field(default_factory=list, description="Explicit deterministic decision criteria applied")


class SemanticImpactAssessment(BaseModel):
    """
    Full semantic impact assessment distinguishing structural reachability from
    demonstrated behavioral, data-flow, control-flow, and contract dependencies.
    """
    changed_symbol: str = Field(..., description="Canonical qualified symbol identity of changed code")
    affected_symbol: str = Field(..., description="Canonical qualified symbol identity of candidate affected symbol")
    structural_depth: int = Field(..., ge=1, description="Graph distance from changed symbol (1 = direct)")
    classification: SemanticImpactClassification = Field(..., description="Semantic classification")
    explanation: str = Field(..., description="Concise explanation of the suspected or established relationship")
    supporting_evidence: list[SemanticEvidenceRef] = Field(default_factory=list, description="Verifiable evidence citations")
    evidence_source: EvidenceSourceType = Field(default=EvidenceSourceType.DIRECTLY_OBSERVED, description="How evidence was produced")
    relevant_tests: list[str] = Field(default_factory=list, description="Tests exercising this specific behavioral relationship")
    test_relevance_reasons: dict[str, str] = Field(default_factory=dict, description="Why each test is considered relevant")
    missing_evidence: list[str] = Field(default_factory=list, description="Missing evidence, unverified assumptions, or limitations")
    uncertainty_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Uncertainty score (0.0 = certain, 1.0 = highly uncertain)")
    analysis_engine: str = Field(default="ast_dataflow", description="Analysis engine or fallback used")
    prioritization: Optional[PrioritizationRank] = Field(default=None, description="Transparent function prioritization rank")
    is_verified: bool = Field(default=True, description="Whether assessment passed hallucination verification")


class EventMatchClassification(str, Enum):
    """
    Evidence-based classification for event-driven test impact relationships:
    - EXPLICIT_EVENT_DISPATCH: Direct, statically identifiable event send (e.g. post_migrate.send).
    - EVENT_TRIGGER_CANDIDATE: Uses trigger infrastructure (e.g. call_command('migrate')) without confirmed behavioral coverage.
    - BEHAVIORAL_COVERAGE: Concrete behavioral evidence connects test to the changed handler.
    - NO_RELEVANT_EVENT_EVIDENCE: No meaningful event or trigger relationship established.
    """
    EXPLICIT_EVENT_DISPATCH = "EXPLICIT_EVENT_DISPATCH"
    EVENT_TRIGGER_CANDIDATE = "EVENT_TRIGGER_CANDIDATE"
    BEHAVIORAL_COVERAGE = "BEHAVIORAL_COVERAGE"
    NO_RELEVANT_EVENT_EVIDENCE = "NO_RELEVANT_EVENT_EVIDENCE"


class EvidenceItem(BaseModel):
    """Individual structured evidence record for explainability in test prioritization."""
    type: str = Field(..., description="Evidence type: COMMAND_DISPATCH, EXPLICIT_DISPATCH, BEHAVIORAL_ASSERTION, DOMAIN_REFERENCE, or DIFF_CO_CHANGE")
    description: str = Field(..., description="Human-readable explanation of this specific evidence piece")
    source_file: str = Field(..., description="Source file where evidence was identified")
    line: Optional[int] = Field(default=None, description="Line number of call or assertion")


class EvolutionRequest(BaseModel):
    base_ref: str = Field(default="main", description="Git base reference (commit SHA, branch, tag)")
    target_ref: Optional[str] = Field(default="HEAD", description="Git target reference (commit SHA, branch, or working tree)")
    repo_path: str = Field(default=".", description="Path to git repository root")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum call-graph traversal depth for indirect impacts")
    enable_semantic_analysis: bool = Field(default=True, description="Whether to run RAG-grounded semantic impact analysis")
    enable_llm_semantic: bool = Field(default=False, description="Whether to invoke local LLM (CodeLlama) in addition to AST data-flow RAG")


class SymbolId(BaseModel):
    """Canonical representation of a Python symbol's qualified identity."""
    file_path: str = Field(..., description="Repository-relative file path")
    name: str = Field(..., description="Function or method name")
    class_name: Optional[str] = Field(default=None, description="Enclosing class name if a method")

    @classmethod
    def get_module_path(cls, file_path: str) -> str:
        p = file_path.replace("\\", "/").strip("./")
        if p.endswith(".py"):
            p = p[:-3]
        if p.endswith("/__init__"):
            p = p[:-9]
        return p.replace("/", ".")

    @property
    def module_path(self) -> str:
        return self.get_module_path(self.file_path)

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.name}" if self.class_name else self.name

    @property
    def canonical_id(self) -> str:
        return f"{self.module_path}.{self.qualified_name}"


class ChangedSymbol(BaseModel):
    name: str = Field(..., description="Function, method, or class name")
    class_name: Optional[str] = Field(default=None, description="Enclosing class name if a method")
    file_path: str = Field(..., description="Repository-relative file path")
    line_start: int = Field(..., description="Starting line in source file")
    line_end: int = Field(..., description="Ending line in source file")
    change_type: ChangeType = Field(default=ChangeType.MODIFIED, description="Type of source change")
    previous_file_path: Optional[str] = Field(default=None, description="Previous file path if symbol or file was renamed")
    parameters: list[str] = Field(default_factory=list, description="Parameter names of the symbol")
    raw_signature: Optional[str] = Field(default=None, description="Stringified signature or declaration")

    @property
    def symbol_name(self) -> str:
        return self.qualified_name

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.name}" if self.class_name else self.name

    @property
    def module_path(self) -> str:
        return SymbolId.get_module_path(self.file_path)

    @property
    def canonical_id(self) -> str:
        return f"{self.module_path}.{self.qualified_name}"

    @property
    def symbol_id(self) -> str:
        return self.canonical_id

    @property
    def is_constructor(self) -> bool:
        return self.name == "__init__" and bool(self.class_name)

    @property
    def start_line(self) -> int:
        return self.line_start

    @property
    def end_line(self) -> int:
        return self.line_end



class EvidenceTrail(BaseModel):
    call_chain: list[str] = Field(default_factory=list, description="Sequence of calls e.g. [target, caller_1, caller_2]")
    caller_file: str = Field(..., description="File where call site is located")
    line_number: Optional[int] = Field(default=None, description="Exact line number of call invocation")
    resolution_engine: str = Field(default="local_ast_fallback", description="Engine used (sourcegraph_graphql or local_ast_fallback)")
    match_quality: str = Field(default="EXACT_AST_CALL", description="Quality of match: EXACT_AST_CALL, ATTRIBUTE_MATCH, or HEURISTIC")
    evidence_source: str = Field(default="AST", description="Human-readable source: AST, Tree-sitter, Sourcegraph, or Fallback AST")
    call_path_description: str = Field(default="", description="Descriptive arrow chain e.g. A -> B -> C")
    is_ambiguous: bool = Field(default=False, description="Whether resolution is ambiguous/unconfirmed")

    @property
    def caller_symbol(self) -> Optional[str]:
        return self.call_chain[-1] if self.call_chain else None

    @property
    def source_symbol(self) -> Optional[str]:
        return self.call_chain[0] if self.call_chain else None



class ImpactNode(BaseModel):
    symbol_name: str = Field(..., description="Name of the affected function or method")
    file_path: str = Field(..., description="File containing the affected symbol")
    line_number: Optional[int] = Field(default=None, description="Line number of definition or call site")
    depth: int = Field(..., ge=1, description="Call-graph distance from changed symbol (1 = direct)")
    impact_type: ImpactType = Field(..., description="DIRECT or INDIRECT impact")
    root_changed_symbol: str = Field(..., description="The original changed symbol that triggered this impact")
    root_changed_file: Optional[str] = Field(default=None, description="The file containing the root changed symbol")
    evidence: EvidenceTrail = Field(..., description="Evidence trail supporting this impact link")
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="Confidence uncertainty (0.0 = completely certain, 1.0 = highly uncertain)")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score (1.0 - uncertainty_score)")
    uncertainty_reason: str = Field(default="", description="Explanation of uncertainty factors")
    is_confirmed: bool = Field(default=True, description="Whether the impact relationship is confirmed vs uncertain heuristic")
    limitations: list[str] = Field(default_factory=list, description="Known architectural or resolution limitations")
    reason: str = Field(..., description="Human-readable explanation of the relationship")
    semantic_assessment: Optional[SemanticImpactAssessment] = Field(
        default=None,
        description="Evidence-grounded semantic impact assessment distinguishing behavioral from structural dependencies",
    )

    @property
    def initiating_symbol(self) -> str:
        return self.root_changed_symbol

    @property
    def initiating_file(self) -> Optional[str]:
        return self.root_changed_file

    @property
    def relationship_type(self) -> str:
        return self.impact_type.value.lower()

    @property
    def semantic_classification(self) -> Optional[SemanticImpactClassification]:
        return self.semantic_assessment.classification if self.semantic_assessment else None



class PrioritizedTest(BaseModel):
    test_name: str = Field(..., description="Test function or method name (e.g., test_apply_valid_and_invalid_coupons)")
    test_file: str = Field(..., description="Relative path to test file")
    line_number: Optional[int] = Field(default=None, description="Starting line number of the test")
    priority_tier: PriorityTier = Field(..., description="Priority tier: CRITICAL, HIGH, MEDIUM, LOW")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Normalized priority ranking score (1.0 = highest)")
    reason: str = Field(..., description="Justification for prioritization")
    selection_reason: str = Field(default="", description="Why this test was prioritized")
    targeted_symbol: str = Field(..., description="The changed or impacted symbol linked to this test")
    targeted_file: Optional[str] = Field(default=None, description="File path of the targeted symbol")
    call_depth: int = Field(default=1, ge=1, description="Graph distance from changed symbol to test")
    impact_distance: str = Field(default="direct", description="Human-readable distance: direct, 1 hop, or multiple hops")
    evidence_type: str = Field(default="confirmed", description="confirmed or heuristic")
    explanation: str = Field(default="", description="Concise developer-facing explanation of selection")
    execution_command: str = Field(default="", description="CLI pytest command to run only this test")
    evidence: Optional[EvidenceTrail] = Field(default=None, description="Detailed evidence chain connecting test to symbol")
    uncertainty_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Uncertainty in test relationship")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score (1.0 - uncertainty_score)")
    match_classification: Optional[EventMatchClassification] = Field(
        default=None,
        description="Event-based match classification (EXPLICIT_EVENT_DISPATCH, EVENT_TRIGGER_CANDIDATE, BEHAVIORAL_COVERAGE, or None)",
    )
    event_name: Optional[str] = Field(default=None, description="Associated event name if matched via event analysis")
    missing_evidence: list[str] = Field(default_factory=list, description="Missing evidence or uncertainty factors")
    structured_evidence: list[EvidenceItem] = Field(default_factory=list, description="Granular structured evidence entries")
    semantic_decision: Optional[str] = Field(default=None, description="HIGH, MEDIUM, or LOW from semantic validation")
    semantic_confidence: Optional[float] = Field(default=None, description="Confidence of semantic validation (0.0 to 1.0)")
    semantic_reason: Optional[str] = Field(default=None, description="Semantic reasoning for behavioral relevance")
    semantic_supporting_evidence: list[str] = Field(default_factory=list, description="Evidence points from CodeLlama/RAG")

    @model_validator(mode="after")
    def populate_defaults(self) -> "PrioritizedTest":
        if not self.selection_reason:
            self.selection_reason = self.reason
        if not self.explanation:
            self.explanation = self.reason
        if not self.execution_command:
            self.execution_command = f"pytest {self.test_file}::{self.test_name} -v"
        if self.evidence is None:
            self.evidence = EvidenceTrail(
                call_chain=[self.test_name, self.targeted_symbol],
                caller_file=self.test_file,
                line_number=self.line_number,
            )
        if self.uncertainty_score is not None and (self.confidence == 0.9 or self.confidence is None):
            self.confidence = round(max(0.0, min(1.0, 1.0 - self.uncertainty_score)), 2)
        return self

    @property
    def file_path(self) -> str:
        return self.test_file

    @property
    def priority(self) -> str:
        return self.priority_tier.value

    @property
    def test_function(self) -> str:
        return self.test_name

    @property
    def target_symbol(self) -> str:
        return self.targeted_symbol

    @property
    def is_direct_caller(self) -> bool:
        return self.call_depth == 1

    @property
    def failure_likelihood(self) -> float:
        return self.priority_score


class EvolutionReport(BaseModel):
    base_ref: str = Field(..., description="Base git reference used for comparison")
    target_ref: str = Field(..., description="Target git reference or working tree")
    diff_stat: str = Field(default="", description="Summary git diff stat")
    changed_files: list[str] = Field(default_factory=list, description="List of files with changes")
    renamed_files: dict[str, str] = Field(default_factory=dict, description="Mapping of previous_path -> new_path for renamed files")
    changed_symbols: list[ChangedSymbol] = Field(default_factory=list, description="Python symbols intersecting diff hunks")
    direct_impacts: list[ImpactNode] = Field(default_factory=list, description="Direct (1-hop) affected callers")
    indirect_impacts: list[ImpactNode] = Field(default_factory=list, description="Transitive (2+ hops) affected callers")
    total_impacted_symbols: int = Field(default=0, description="Total count of unique impacted symbols")
    prioritized_tests: list[PrioritizedTest] = Field(default_factory=list, description="Ranked list of existing tests to execute")
    event_trigger_candidates: list[PrioritizedTest] = Field(
        default_factory=list,
        description="Candidate tests that invoke event triggers without confirmed behavioral coverage",
    )
    insufficient_evidence_tests: list[PrioritizedTest] = Field(
        default_factory=list,
        description="Tests inspected with insufficient evidence to recommend",
    )
    total_discovered_tests: int = Field(
        default=0,
        description="Total test functions/methods discovered across the repository",
    )
    analysis_latency_ms: float = Field(default=0.0, description="Analysis execution latency in milliseconds")
    warnings: list[str] = Field(default_factory=list, description="Warnings or limitations encountered during analysis")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Precision, recall, and evaluation metadata")
    semantic_impacts: list[SemanticImpactAssessment] = Field(
        default_factory=list,
        description="Evidence-grounded semantic impact assessments distinguishing behavioral from structural dependencies",
    )
    semantic_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary counts and metric distribution of semantic impact classifications",
    )

    @property
    def total_files_changed(self) -> int:
        return len(self.changed_files)

    @property
    def direct_impact_count(self) -> int:
        return len(self.direct_impacts)

    @property
    def indirect_impact_count(self) -> int:
        return len(self.indirect_impacts)

    @property
    def impact_graph(self) -> list[ImpactNode]:
        return self.direct_impacts + self.indirect_impacts

    @property
    def direct_behavioral_impacts(self) -> list[SemanticImpactAssessment]:
        return [
            s for s in self.semantic_impacts
            if s.classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY
        ]

    @property
    def indirect_behavioral_impacts(self) -> list[SemanticImpactAssessment]:
        return [
            s for s in self.semantic_impacts
            if s.classification == SemanticImpactClassification.INDIRECT_BEHAVIORAL_DEPENDENCY
        ]

    @property
    def structural_only_impacts(self) -> list[SemanticImpactAssessment]:
        return [
            s for s in self.semantic_impacts
            if s.classification == SemanticImpactClassification.STRUCTURAL_ONLY
        ]

    @property
    def uncertain_impacts(self) -> list[SemanticImpactAssessment]:
        return [
            s for s in self.semantic_impacts
            if s.classification == SemanticImpactClassification.UNCERTAIN
        ]

    @property
    def prioritized_test_count(self) -> int:
        return len(self.prioritized_tests)

    @property
    def event_candidate_count(self) -> int:
        return len(self.event_trigger_candidates)

    @property
    def critical_tests(self) -> list[PrioritizedTest]:
        return [t for t in self.prioritized_tests if t.priority_tier == PriorityTier.CRITICAL]

    @property
    def high_priority_tests(self) -> list[PrioritizedTest]:
        return [t for t in self.prioritized_tests if t.priority_tier == PriorityTier.HIGH]

    @property
    def behavior_supported_tests(self) -> list[PrioritizedTest]:
        return [
            t for t in self.prioritized_tests
            if t.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE
        ]

    @property
    def directly_supported_tests(self) -> list[PrioritizedTest]:
        return [
            t for t in self.prioritized_tests
            if t.call_depth == 1 and t.evidence_type == "confirmed"
        ]

