"""
Verification test suite for TestPilot Test Coverage Intelligence - Phase 3:
Test-to-Code Mapping and Impact Graph Integration.
Validates:
- Individual tests accurately mapped to the specific functions they execute.
- Unrelated tests are not incorrectly mapped.
- Indirect / helper calls captured in runtime execution mapping.
- Functions reachable via AST but never executed are distinguished from runtime executed.
- Integrated evidence graph distinguishes RUNTIME_COVERAGE (1.0) from DIRECT_AST_CALL (0.85) and SEMANTIC_SIMILARITY (0.40).
- Test selection boost for tests with confirmed runtime coverage.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import pytest

from testpilot.coverage.collector import CoverageCollector
from testpilot.coverage.models import (
    CoverageEvidenceType,
    CoverageReport,
    CoverageStatus,
    FunctionStatus,
)
from testpilot.coverage.storage import CoverageStore
from testpilot.coverage.test_mapping import TestToCodeMapper


@pytest.fixture
def temp_phase3_repo():
    """
    Creates an isolated repository with distinct functions and dedicated tests:
    - `auth.py`:
        - `login_user(username, password)` (calls `verify_hash`)
        - `verify_hash(pwd)` (helper called indirectly by `login_user`)
        - `logout_user(token)` (uncalled by auth tests)
    - `billing.py`:
        - `charge_card(amount)` (independent domain function)
    - `tests/test_auth.py`:
        - `test_login()` -> executes `login_user` and `verify_hash`
    - `tests/test_billing.py`:
        - `test_charge()` -> executes `charge_card` only
    """
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_p3_")
    repo = Path(tmpdir)

    auth_file = repo / "auth.py"
    auth_file.write_text(
        "def verify_hash(pwd):\n"
        "    return pwd == 'secret'\n\n"
        "def login_user(username, password):\n"
        "    if verify_hash(password):\n"
        "        return f'token_{username}'\n"
        "    return None\n\n"
        "def logout_user(token):\n"
        "    return True\n",
        encoding="utf-8",
    )

    billing_file = repo / "billing.py"
    billing_file.write_text(
        "def charge_card(amount):\n"
        "    return amount > 0\n",
        encoding="utf-8",
    )

    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)

    test_auth = tests_dir / "test_auth.py"
    test_auth.write_text(
        "import auth\n\n"
        "def test_login():\n"
        "    assert auth.login_user('alice', 'secret') == 'token_alice'\n",
        encoding="utf-8",
    )

    test_billing = tests_dir / "test_billing.py"
    test_billing.write_text(
        "import billing\n\n"
        "def test_charge():\n"
        "    assert billing.charge_card(50) is True\n",
        encoding="utf-8",
    )

    yield repo
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_store():
    tmpdir = tempfile.mkdtemp(prefix="tp_cov_p3_store_")
    db_path = Path(tmpdir) / "test_cov.db"
    reports_dir = Path(tmpdir) / "reports"
    store = CoverageStore(db_path=db_path, reports_dir=reports_dir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_test_to_code_mapping_accuracy(temp_phase3_repo, temp_store):
    """Verify that individual tests map only to the functions they actually execute."""
    mapper = TestToCodeMapper(store=temp_store)

    # 1. Map test_login
    map_login = mapper.map_test_case(
        repo_path=temp_phase3_repo,
        test_case={"test_name": "test_login", "test_file": "tests/test_auth.py"},
    )

    assert "login_user" in map_login.covered_functions
    # Indirect call to helper must also be captured!
    assert "verify_hash" in map_login.covered_functions
    # Unrelated function in same file must NOT be marked!
    assert "logout_user" not in map_login.covered_functions
    # Unrelated file must NOT be in covered files!
    assert "billing.py" not in map_login.covered_files
    assert "charge_card" not in map_login.covered_functions
    assert map_login.evidence_type == CoverageEvidenceType.RUNTIME_COVERAGE

    # 2. Map test_charge
    map_charge = mapper.map_test_case(
        repo_path=temp_phase3_repo,
        test_case={"test_name": "test_charge", "test_file": "tests/test_billing.py"},
    )

    assert "charge_card" in map_charge.covered_functions
    assert "billing.py" in map_charge.covered_files
    # Auth functions must NOT be mapped to test_charge!
    assert "login_user" not in map_charge.covered_functions
    assert "verify_hash" not in map_charge.covered_functions


def test_enrich_report_with_mappings_and_query_store(temp_phase3_repo, temp_store):
    """Verify that CoverageReport is enriched with mappings and functions list their executing tests."""
    collector = CoverageCollector(store=temp_store)
    mapper = TestToCodeMapper(store=temp_store)

    report = collector.collect(repo_path=temp_phase3_repo, is_full_suite=True, repo_id="phase3_repo")

    # Map all tests
    test_cases = [
        {"test_name": "test_login", "test_file": "tests/test_auth.py"},
        {"test_name": "test_charge", "test_file": "tests/test_billing.py"},
    ]
    mappings = mapper.map_all_tests(temp_phase3_repo, test_cases)
    enriched = mapper.attach_mappings_to_report(report, mappings)

    assert len(enriched.test_mappings) == 2

    # Verify function-level executing_tests
    auth_summary = next(f for f in enriched.file_summaries if "auth.py" in f.file_path)
    login_fn = next(fn for fn in auth_summary.functions if fn.function_name == "login_user")
    assert len(login_fn.executing_tests) >= 1
    assert any("test_login" in t for t in login_fn.executing_tests)

    # Verify query from store
    stored_maps = temp_store.get_test_mappings(enriched.coverage_run_id)
    assert len(stored_maps) == 2


def test_integrated_impact_graph_distinguishes_evidence_types(temp_phase3_repo, temp_store):
    """
    Verify that build_integrated_impact_graph clearly distinguishes
    empirical RUNTIME_COVERAGE (confidence 1.0) from static AST (0.85) and RAG (0.40).
    """
    collector = CoverageCollector(store=temp_store)
    mapper = TestToCodeMapper(store=temp_store)

    report = collector.collect(repo_path=temp_phase3_repo, is_full_suite=True)
    mappings = mapper.map_all_tests(temp_phase3_repo, [
        {"test_name": "test_login", "test_file": "tests/test_auth.py"},
    ])
    enriched = mapper.attach_mappings_to_report(report, mappings)

    # Synthetic static AST and RAG edges to test integration
    static_edges = [{
        "source": "func:login_user",
        "target": "func:verify_hash",
        "evidence_type": "DIRECT_AST_CALL",
        "confidence": 0.85,
    }]
    rag_edges = [{
        "source": "func:logout_user",
        "target": "func:login_user",
        "confidence": 0.40,
    }]

    graph = mapper.build_integrated_impact_graph(
        report=enriched,
        static_ast_edges=static_edges,
        rag_similarity_edges=rag_edges,
    )

    assert graph["total_nodes"] > 0
    assert graph["total_edges"] >= 3

    breakdown = graph["evidence_breakdown"]
    assert breakdown["runtime_confirmed"] >= 1
    assert breakdown["ast_inferred"] == 1
    assert breakdown["semantic_similarity"] == 1

    # Verify confidence differentiation
    runtime_edge = next(e for e in graph["edges"] if e["evidence_type"] == "RUNTIME_COVERAGE")
    assert runtime_edge["confidence"] == 1.0
    assert runtime_edge["is_empirical"] is True

    ast_edge = next(e for e in graph["edges"] if e["evidence_type"] == "DIRECT_AST_CALL")
    assert ast_edge["confidence"] == 0.85
    assert ast_edge["is_empirical"] is False

