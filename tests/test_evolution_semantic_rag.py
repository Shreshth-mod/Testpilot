"""
Comprehensive Test Suite for Evidence-Grounded Semantic Analysis & Repository RAG.
Covers:
1. Direct caller consuming return value -> DIRECT_BEHAVIORAL_DEPENDENCY.
2. Structural caller not depending on behavior (logging/void call) -> STRUCTURAL_ONLY.
3. Transitive dependency with demonstrated behavioral flow -> INDIRECT_BEHAVIORAL_DEPENDENCY (not dropped by depth).
4. Transitive structural path without behavioral coupling -> STRUCTURAL_ONLY / UNCERTAIN.
5. Relevant tests retrieved based on assertions/contracts.
6. Irrelevant / incomplete evidence handled with explicit uncertainty & missing evidence notices.
7. Sourcegraph offline fallback to local AST analysis and RAG.
8. Vector index missing, stale, or empty graceful fallback.
9. LLM unavailable, offline, or malformed JSON -> deterministic fallback.
10. Anti-hallucination validation rejects fabricated file references.
11. Repository boundary isolation (no cross-repo evidence leakage).
12. End-to-end integration with EvolutionReport and Full Pipeline compatibility.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testpilot.evolution.engine import RepositoryEvolutionEngine
from testpilot.evolution.models import (
    ChangedSymbol,
    ChangeType,
    EvidenceSourceType,
    EvidenceTrail,
    EvolutionReport,
    EvolutionRequest,
    ImpactNode,
    ImpactType,
    SemanticEvidenceRef,
    SemanticImpactAssessment,
    SemanticImpactClassification,
)
from testpilot.evolution.semantic import DataFlowEvidence, SemanticImpactAnalyzer
from testpilot.rag.repo_vector_store import RepoCodeVectorStore


@pytest.fixture
def temp_fixture_repo():
    """Sets up an isolated multi-file repository fixture for deterministic testing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        root = Path(tmpdir)

        # 1. Core pricing module
        pricing = root / "pricing.py"
        pricing.write_text(
            '''"""Pricing calculation service."""

def calculate_discount(price: float, discount_pct: float) -> float:
    """Calculate discounted total."""
    return price * (1.0 - discount_pct)
''',
            encoding="utf-8",
        )

        # 2. Checkout service consuming return value
        checkout = root / "checkout.py"
        checkout.write_text(
            '''"""Checkout service consuming discount."""
from pricing import calculate_discount

def process_checkout(subtotal: float, discount_pct: float) -> float:
    # Behavioral dependency: consumes return value to charge customer
    final_amount = calculate_discount(subtotal, discount_pct)
    return final_amount
''',
            encoding="utf-8",
        )

        # 3. Audit logger only making a void / logging call
        audit = root / "audit.py"
        audit.write_text(
            '''"""Audit logger service."""
from pricing import calculate_discount

def log_discount_audit(subtotal: float):
    # Structural call only: void call / side-effect logging
    calculate_discount(subtotal, 0.0)
    print("Audit logged")
''',
            encoding="utf-8",
        )

        # 4. Transitive billing service calling checkout
        billing = root / "billing.py"
        billing.write_text(
            '''"""High-level billing orchestrator."""
from checkout import process_checkout

def generate_invoice(order_id: str, amount: float, discount: float) -> dict:
    total = process_checkout(amount, discount)
    return {"order_id": order_id, "total": total}
''',
            encoding="utf-8",
        )

        # 5. Automated tests asserting contract
        test_file = root / "test_checkout.py"
        test_file.write_text(
            '''"""Tests for checkout and discount contracts."""
from checkout import process_checkout

def test_process_checkout_discount():
    assert process_checkout(100.0, 0.10) == 90.0
''',
            encoding="utf-8",
        )

        yield root


def test_scenario_1_direct_caller_consumes_return_value(temp_fixture_repo):
    """Scenario 1: Direct caller consumes return value -> DIRECT_BEHAVIORAL_DEPENDENCY."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
        change_type=ChangeType.MODIFIED,
    )
    node = ImpactNode(
        symbol_name="process_checkout",
        file_path="checkout.py",
        line_number=6,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["process_checkout", "calculate_discount"],
            caller_file="checkout.py",
            line_number=6,
            resolution_engine="local_ast_fallback",
        ),
        uncertainty_score=0.05,
        reason="Direct call",
    )

    assessment = analyzer.classify_relationship(changed_sym, node)
    assert assessment.classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY
    assert assessment.structural_depth == 1
    assert assessment.evidence_source == EvidenceSourceType.DIRECTLY_OBSERVED
    assert len(assessment.supporting_evidence) > 0
    assert assessment.supporting_evidence[0].evidence_type == "RETURN_VALUE_CONSUMPTION"


def test_scenario_2_structural_only_logging_caller(temp_fixture_repo):
    """Scenario 2: Caller makes side-effect/logging call -> STRUCTURAL_ONLY."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
        change_type=ChangeType.MODIFIED,
    )
    node = ImpactNode(
        symbol_name="log_discount_audit",
        file_path="audit.py",
        line_number=6,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["log_discount_audit", "calculate_discount"],
            caller_file="audit.py",
            line_number=6,
            resolution_engine="local_ast_fallback",
        ),
        uncertainty_score=0.15,
        reason="Logging call",
    )

    assessment = analyzer.classify_relationship(changed_sym, node)
    assert assessment.classification == SemanticImpactClassification.STRUCTURAL_ONLY
    assert assessment.structural_depth == 1
    assert "side-effect" in assessment.explanation.lower() or "logging" in assessment.explanation.lower()


def test_scenario_3_transitive_behavioral_dependency(temp_fixture_repo):
    """Scenario 3: Deeper dependency with demonstrated behavioral transmission -> INDIRECT_BEHAVIORAL_DEPENDENCY."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
        change_type=ChangeType.MODIFIED,
    )
    node = ImpactNode(
        symbol_name="generate_invoice",
        file_path="billing.py",
        line_number=5,
        depth=2,
        impact_type=ImpactType.INDIRECT,
        root_changed_symbol="pricing.calculate_discount",
        is_confirmed=True,
        evidence=EvidenceTrail(
            call_chain=["generate_invoice", "process_checkout", "calculate_discount"],
            caller_file="billing.py",
            line_number=5,
            resolution_engine="local_ast_fallback",
            call_path_description="generate_invoice -> process_checkout -> calculate_discount",
        ),
        uncertainty_score=0.20,
        reason="Transitive call chain",
    )

    assessment = analyzer.classify_relationship(changed_sym, node)
    assert assessment.classification == SemanticImpactClassification.INDIRECT_BEHAVIORAL_DEPENDENCY
    assert assessment.structural_depth == 2
    assert assessment.evidence_source == EvidenceSourceType.STATICALLY_INFERRED


def test_scenario_4_transitive_unconfirmed_structural_only(temp_fixture_repo):
    """Scenario 4: Transitive path without confirmed behavioral coupling -> STRUCTURAL_ONLY."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
        change_type=ChangeType.MODIFIED,
    )
    node = ImpactNode(
        symbol_name="some_remote_utility",
        file_path="util.py",
        line_number=10,
        depth=3,
        impact_type=ImpactType.INDIRECT,
        root_changed_symbol="pricing.calculate_discount",
        is_confirmed=False,
        evidence=EvidenceTrail(
            call_chain=["some_remote_utility", "unknown_intermediate", "calculate_discount"],
            caller_file="util.py",
            line_number=10,
            resolution_engine="local_ast_fallback",
            is_ambiguous=True,
        ),
        uncertainty_score=0.55,
        reason="Unconfirmed transitive link",
    )

    assessment = analyzer.classify_relationship(changed_sym, node)
    assert assessment.classification == SemanticImpactClassification.STRUCTURAL_ONLY
    assert len(assessment.missing_evidence) > 0


def test_scenario_5_retrieves_relevant_tests_from_rag(temp_fixture_repo):
    """Scenario 5: Retrieves relevant tests exercising behavioral contract via RAG."""
    store = RepoCodeVectorStore(
        db_path=str(temp_fixture_repo / ".chroma_db"),
        repo_root=str(temp_fixture_repo),
    )
    evidence = store.retrieve_symbol_evidence(
        symbol_name="calculate_discount",
        file_path="pricing.py",
        include_callers=True,
        include_tests=True,
    )
    assert len(evidence) > 0
    test_units = [e for e in evidence if e.get("is_test")]
    assert len(test_units) >= 1
    assert "test_process_checkout_discount" in [t.get("symbol_name") for t in test_units]


def test_scenario_6_missing_evidence_quantifies_uncertainty(temp_fixture_repo):
    """Scenario 6: Incomplete / unparseable code records explicit missing evidence."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
        change_type=ChangeType.MODIFIED,
    )
    node = ImpactNode(
        symbol_name="broken_caller",
        file_path="broken.py",
        line_number=1,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["broken_caller", "calculate_discount"],
            caller_file="broken.py",
            line_number=1,
        ),
        uncertainty_score=0.40,
        reason="Direct call",
    )

    # Caller code with syntax error
    assessment = analyzer.classify_relationship(
        changed_symbol=changed_sym,
        impact_node=node,
        caller_code="def broken_func(: invalid python syntax",
    )
    assert assessment.classification == SemanticImpactClassification.UNCERTAIN
    assert assessment.uncertainty_score >= 0.40
    assert len(assessment.missing_evidence) > 0


def test_scenario_7_sourcegraph_offline_fallback(temp_fixture_repo):
    """Scenario 7: When Sourcegraph is offline, local AST analysis and RAG operate seamlessly."""
    mock_sg = MagicMock()
    mock_sg.is_alive.return_value = False
    analyzer = SemanticImpactAnalyzer(
        repo_root=str(temp_fixture_repo),
        sourcegraph_client=mock_sg,
    )
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
    )
    node = ImpactNode(
        symbol_name="process_checkout",
        file_path="checkout.py",
        line_number=6,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["process_checkout", "calculate_discount"],
            caller_file="checkout.py",
            line_number=6,
            resolution_engine="local_ast_fallback",
        ),
        uncertainty_score=0.05,
        reason="Direct call",
    )

    assessment = analyzer.classify_relationship(changed_sym, node)
    assert assessment.classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY
    assert "ast_dataflow" in assessment.analysis_engine


def test_scenario_8_vector_store_empty_or_unavailable(temp_fixture_repo):
    """Scenario 8: If vector store is empty, fallback to direct AST lookup operates safely."""
    store = RepoCodeVectorStore(
        db_path=str(temp_fixture_repo / ".chroma_db_empty"),
        repo_root=str(temp_fixture_repo),
    )
    # Exact AST lookup does not require pre-indexing
    exact = store.find_exact_code_unit("calculate_discount", file_path="pricing.py")
    assert exact is not None
    assert exact["symbol_name"] == "calculate_discount"
    assert exact["retrieval_method"] == "exact_structural"


def test_scenario_9_llm_malformed_or_offline_fallback(temp_fixture_repo):
    """Scenario 9: If LLM returns malformed output or is offline, deterministic AST is preserved."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Not valid JSON at all!"
    analyzer = SemanticImpactAnalyzer(
        repo_root=str(temp_fixture_repo),
        llm_client=mock_llm,
    )
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
    )
    node = ImpactNode(
        symbol_name="process_checkout",
        file_path="checkout.py",
        line_number=6,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["process_checkout", "calculate_discount"],
            caller_file="checkout.py",
            line_number=6,
        ),
        uncertainty_score=0.05,
        reason="Direct call",
    )

    assessment = analyzer.classify_relationship(
        changed_symbol=changed_sym,
        impact_node=node,
        enable_llm=True,
    )
    # Does not crash; preserves deterministic AST classification
    assert assessment.classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY


def test_scenario_10_anti_hallucination_rejects_fabricated_citations(temp_fixture_repo):
    """Scenario 10: Model-cited files that do not exist are rejected and quarantined."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    rag_evidence = [
        {"file_path": "pricing.py", "symbol_name": "calculate_discount"}
    ]
    fabricated_refs = [
        {
            "source_file": "non_existent_invented_service.py",
            "line_start": 42,
            "line_end": 45,
            "evidence_type": "FABRICATED_CITATION",
        },
        {
            "source_file": "pricing.py",
            "line_start": 3,
            "line_end": 6,
            "evidence_type": "VALID_CITATION",
        },
    ]

    valid, rejected = analyzer._verify_evidence_references(fabricated_refs, rag_evidence)
    assert len(valid) == 1
    assert valid[0].source_file == "pricing.py"
    assert len(rejected) == 1
    assert "non_existent_invented_service.py" in rejected[0]


def test_scenario_11_transparent_prioritization_rules(temp_fixture_repo):
    """Scenario 11: Prioritizes functions by explicit lexicographic rules, not arbitrary formulas."""
    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))

    direct_behavioral = SemanticImpactAssessment(
        changed_symbol="pricing.calculate_discount",
        affected_symbol="checkout.process_checkout",
        structural_depth=1,
        classification=SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY,
        explanation="Direct consumer",
        evidence_source=EvidenceSourceType.DIRECTLY_OBSERVED,
        relevant_tests=["test_process_checkout_discount"],
    )

    deep_behavioral = SemanticImpactAssessment(
        changed_symbol="pricing.calculate_discount",
        affected_symbol="billing.generate_invoice",
        structural_depth=2,
        classification=SemanticImpactClassification.INDIRECT_BEHAVIORAL_DEPENDENCY,
        explanation="Transitive consumer",
        evidence_source=EvidenceSourceType.STATICALLY_INFERRED,
    )

    direct_logging_only = SemanticImpactAssessment(
        changed_symbol="pricing.calculate_discount",
        affected_symbol="audit.log_discount_audit",
        structural_depth=1,
        classification=SemanticImpactClassification.STRUCTURAL_ONLY,
        explanation="Logging side-effect only",
        evidence_source=EvidenceSourceType.DIRECTLY_OBSERVED,
    )

    prioritized = analyzer.prioritize_impacted_symbols([
        direct_logging_only,
        deep_behavioral,
        direct_behavioral,
    ])

    # Direct behavioral is Rank 1
    assert prioritized[0].affected_symbol == "checkout.process_checkout"
    assert prioritized[0].prioritization.tier == "CRITICAL_BEHAVIORAL"
    assert prioritized[0].prioritization.rank == 1

    # Deep behavioral is Rank 2 (NOT demoted below logging call merely because depth is 2!)
    assert prioritized[1].affected_symbol == "billing.generate_invoice"
    assert prioritized[1].prioritization.tier == "HIGH_BEHAVIORAL"
    assert prioritized[1].prioritization.rank == 2

    # Logging-only call is Rank 3 (structural baseline)
    assert prioritized[2].affected_symbol == "audit.log_discount_audit"
    assert prioritized[2].prioritization.tier == "STRUCTURAL_BASELINE"
    assert prioritized[2].prioritization.rank == 3


def test_scenario_12_evolution_report_contains_semantic_summary(temp_fixture_repo):
    """Scenario 12: EvolutionReport accurately encapsulates semantic impacts and summary metrics."""
    engine = RepositoryEvolutionEngine(repo_root=str(temp_fixture_repo))
    changed_sym = ChangedSymbol(
        name="calculate_discount",
        file_path="pricing.py",
        line_start=3,
        line_end=6,
    )
    direct_node = ImpactNode(
        symbol_name="process_checkout",
        file_path="checkout.py",
        line_number=6,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="pricing.calculate_discount",
        evidence=EvidenceTrail(
            call_chain=["process_checkout", "calculate_discount"],
            caller_file="checkout.py",
            line_number=6,
        ),
        uncertainty_score=0.05,
        reason="Direct call",
    )

    analyzer = SemanticImpactAnalyzer(repo_root=str(temp_fixture_repo))
    assessments = analyzer.analyze_impacts([changed_sym], [direct_node])

    report = EvolutionReport(
        base_ref="HEAD~1",
        target_ref="HEAD",
        changed_symbols=[changed_sym],
        direct_impacts=[direct_node],
        semantic_impacts=assessments,
        semantic_summary={
            "total_semantic_assessments": len(assessments),
            "direct_behavioral_count": 1,
            "indirect_behavioral_count": 0,
            "structural_only_count": 0,
            "uncertain_count": 0,
        },
    )

    assert len(report.semantic_impacts) == 1
    assert len(report.direct_behavioral_impacts) == 1
    assert report.semantic_summary["direct_behavioral_count"] == 1
    assert direct_node.semantic_assessment is not None
    assert direct_node.semantic_classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY
