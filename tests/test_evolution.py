"""
Tests for Repository Evolution Intelligence.
Validates git diff extraction, AST changed symbol resolution, multi-hop transitive blast radius,
test prioritization with failure-likelihood ranking, and FastAPI web endpoints.
"""

import subprocess

from fastapi.testclient import TestClient

from testpilot.evolution.engine import RepositoryEvolutionEngine
from testpilot.evolution.models import (
    ChangedSymbol,
    ChangeType,
    EvidenceTrail,
    EvolutionReport,
    ImpactNode,
    ImpactType,
    PrioritizedTest,
    PriorityTier,
)
from testpilot.web.api import app


def test_evolution_models():
    """Verify data model construction, validations, and convenience properties."""
    sym = ChangedSymbol(
        name="calculate_order_totals",
        file_path="testbed/app/services/order_service.py",
        line_start=30,
        line_end=65,
        change_type=ChangeType.MODIFIED,
    )
    assert sym.symbol_name == "calculate_order_totals"
    assert sym.start_line == 30
    assert sym.end_line == 65

    evidence = EvidenceTrail(
        call_chain=["calculate_order_totals", "create_order"],
        caller_file="testbed/app/routers/orders.py",
        line_number=45,
    )
    assert evidence.source_symbol == "calculate_order_totals"
    assert evidence.caller_symbol == "create_order"

    node = ImpactNode(
        symbol_name="create_order",
        file_path="testbed/app/routers/orders.py",
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="calculate_order_totals",
        evidence=evidence,
        uncertainty_score=0.1,
        reason="Direct caller invocation",
    )
    assert node.depth == 1
    assert node.impact_type == ImpactType.DIRECT

    test_item = PrioritizedTest(
        test_name="test_order_total_calculation",
        test_file="tests/test_order_service.py",
        line_number=10,
        priority_tier=PriorityTier.CRITICAL,
        priority_score=0.95,
        reason="Direct call to modified calculate_order_totals",
        targeted_symbol="calculate_order_totals",
        call_depth=1,
        execution_command="pytest tests/test_order_service.py::test_order_total_calculation -v",
        evidence=evidence,
    )
    assert test_item.test_function == "test_order_total_calculation"
    assert test_item.target_symbol == "calculate_order_totals"
    assert test_item.is_direct_caller is True
    assert test_item.failure_likelihood == 0.95

    report = EvolutionReport(
        base_ref="HEAD~1",
        target_ref="HEAD",
        diff_stat="1 file changed, 10 insertions(+)",
        changed_files=["testbed/app/services/order_service.py"],
        changed_symbols=[sym],
        direct_impacts=[node],
        indirect_impacts=[],
        total_impacted_symbols=1,
        prioritized_tests=[test_item],
        analysis_latency_ms=12.5,
    )
    assert report.total_files_changed == 1
    assert report.direct_impact_count == 1
    assert report.indirect_impact_count == 0
    assert len(report.impact_graph) == 1
    assert report.prioritized_test_count == 1
    assert len(report.critical_tests) == 1
    assert len(report.high_priority_tests) == 0


def test_resolve_git_diff():
    """Verify git diff retrieval runs non-destructively and handles commits."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    diff_text, changed_files, diff_stat = engine.resolve_git_diff("HEAD~1", "HEAD")
    assert isinstance(diff_text, str)
    assert isinstance(changed_files, list)
    assert isinstance(diff_stat, str)


def test_extract_changed_symbols_from_diff():
    """Verify AST hunk intersection accurately identifies modified function symbols."""
    engine = RepositoryEvolutionEngine(repo_root=".")

    sample_diff = """diff --git a/testbed/app/services/order_service.py b/testbed/app/services/order_service.py
index abc1234..def5678 100644
--- a/testbed/app/services/order_service.py
+++ b/testbed/app/services/order_service.py
@@ -82,6 +82,7 @@ def calculate_order_totals(cls, items, coupon_code=None):
+    # Modified coupon discount calculation
"""
    symbols = engine.extract_changed_symbols(sample_diff, ["testbed/app/services/order_service.py"])
    assert len(symbols) >= 1
    matched = [s for s in symbols if s.name == "calculate_order_totals"]
    assert len(matched) == 1
    assert matched[0].file_path == "testbed/app/services/order_service.py"
    assert matched[0].change_type == ChangeType.MODIFIED



def test_build_transitive_impact_graph():
    """Verify multi-hop transitive BFS builds depth-tagged impact graph with cycle avoidance."""
    engine = RepositoryEvolutionEngine(repo_root=".")

    changed_symbols = [
        ChangedSymbol(
            name="calculate_order_totals",
            file_path="testbed/app/services/order_service.py",
            line_start=30,
            line_end=70,
            change_type=ChangeType.MODIFIED,
        )
    ]

    direct_impacts, indirect_impacts = engine.build_transitive_impact_graph(changed_symbols, max_depth=3)

    # Direct impacts must be depth 1
    for d in direct_impacts:
        assert d.depth == 1
        assert d.impact_type == ImpactType.DIRECT
        assert d.root_changed_symbol == "calculate_order_totals"

    # Indirect impacts must be depth >= 2
    for ind in indirect_impacts:
        assert ind.depth >= 2
        assert ind.impact_type == ImpactType.INDIRECT
        assert ind.root_changed_symbol == "calculate_order_totals"


def test_prioritize_tests():
    """Verify test prioritization ranks direct callers higher than indirect callers."""
    engine = RepositoryEvolutionEngine(repo_root=".")

    changed_symbols = [
        ChangedSymbol(
            name="calculate_order_totals",
            file_path="testbed/app/services/order_service.py",
            line_start=30,
            line_end=70,
            change_type=ChangeType.MODIFIED,
        )
    ]
    direct_impacts, indirect_impacts = engine.build_transitive_impact_graph(changed_symbols, max_depth=3)

    tests = engine.prioritize_tests(changed_symbols, direct_impacts, indirect_impacts)
    assert isinstance(tests, list)

    if tests:
        # Check ordering: priority scores must be monotonically non-increasing
        scores = [t.priority_score for t in tests]
        assert scores == sorted(scores, reverse=True)

        # Check execution command format
        for t in tests:
            assert t.execution_command.startswith("pytest ")
            assert "::" in t.execution_command or ".py" in t.execution_command


def test_uncertainty_score():
    """Verify uncertainty score penalizes deeper graph hops and ambiguous names."""
    engine = RepositoryEvolutionEngine(repo_root=".")

    score_hop1 = engine._compute_uncertainty_score(1, "calculate_order_totals", "sourcegraph_graphql")
    score_hop2 = engine._compute_uncertainty_score(2, "calculate_order_totals", "sourcegraph_graphql")
    score_hop3 = engine._compute_uncertainty_score(3, "calculate_order_totals", "sourcegraph_graphql")

    assert score_hop1 < score_hop2 < score_hop3

    # Ambiguous generic names should have higher uncertainty
    score_generic = engine._compute_uncertainty_score(1, "run", "sourcegraph_graphql")
    assert score_generic > score_hop1


def test_api_evolution_endpoints():
    """Verify FastAPI evolution REST endpoints."""
    client = TestClient(app)

    # 1. GET /api/evolution/refs
    refs_res = client.get("/api/evolution/refs")
    assert refs_res.status_code == 200
    refs_data = refs_res.json()
    assert "current_branch" in refs_data
    assert "branches" in refs_data
    assert "recent_commits" in refs_data
    assert "presets" in refs_data

    # 2. POST /api/evolution/analyze
    analyze_payload = {
        "base_ref": "HEAD~1",
        "target_ref": "HEAD",
        "max_depth": 2,
        "repo_path": ".",
    }
    res = client.post("/api/evolution/analyze", json=analyze_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["base_ref"] == "HEAD~1"
    assert data["target_ref"] == "HEAD"
    assert "changed_files" in data
    assert "changed_symbols" in data
    assert "direct_impacts" in data
    assert "indirect_impacts" in data
    assert "prioritized_tests" in data
    assert "analysis_latency_ms" in data
    assert data["analysis_latency_ms"] >= 0.0
    assert "warnings" in data
    assert isinstance(data["warnings"], list)


def test_evolution_evaluator_benchmark():
    """Verify empirical evaluation benchmark measures precision, recall, and F1."""
    from testpilot.evolution.evaluator import EvolutionEvaluator

    evaluator = EvolutionEvaluator()
    summary = evaluator.run_all()

    assert summary["total_scenarios"] >= 3
    assert summary["mean_recall"] >= 0.80
    assert summary["mean_precision"] >= 0.70
    assert summary["mean_f1_score"] >= 0.75
    assert summary["mean_latency_ms"] > 0.0


def test_evolution_warnings_and_fallback():
    """Verify warnings are generated when Sourcegraph is in fallback mode or non-py files change."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    report = engine.analyze()

    assert isinstance(report.warnings, list)
    # Since Sourcegraph daemon is typically offline in local dev, warning must be captured
    assert any("Sourcegraph" in w for w in report.warnings)


def test_invalid_git_ref_handling():
    """Verify engine rejects invalid/non-existent git refs explicitly by raising InvalidGitReferenceError."""
    import pytest

    from testpilot.evolution.engine import InvalidGitReferenceError

    engine = RepositoryEvolutionEngine(repo_root=".")

    # Invalid base ref
    with pytest.raises(InvalidGitReferenceError) as exc_base:
        engine.resolve_git_diff("non_existent_ref_xyz", "HEAD")
    assert "non_existent_ref_xyz" in str(exc_base.value)

    # Invalid target ref
    with pytest.raises(InvalidGitReferenceError) as exc_target:
        engine.resolve_git_diff("HEAD", "non_existent_target_abc")
    assert "non_existent_target_abc" in str(exc_target.value)


def test_api_invalid_git_ref_returns_400():
    """Verify API returns HTTP 400 with safe error message when invalid Git reference is supplied."""
    client = TestClient(app)

    # Invalid base ref
    res1 = client.post("/api/evolution/analyze", json={"base_ref": "invalid_branch_xyz", "target_ref": "HEAD"})
    assert res1.status_code == 400
    assert "invalid_branch_xyz" in res1.json()["detail"]
    assert "Traceback" not in res1.json()["detail"]

    # Invalid target ref
    res2 = client.post("/api/evolution/analyze", json={"base_ref": "HEAD", "target_ref": "invalid_target_123"})
    assert res2.status_code == 400
    assert "invalid_target_123" in res2.json()["detail"]
    assert "Traceback" not in res2.json()["detail"]

    # Valid HEAD~1..HEAD
    res3 = client.post("/api/evolution/analyze", json={"base_ref": "HEAD~1", "target_ref": "HEAD"})
    assert res3.status_code == 200


def test_cli_invalid_git_ref_exits_nonzero():
    """Verify CLI returns exit code 1 with clear error message when invalid ref is passed."""
    from typer.testing import CliRunner

    from testpilot.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["evolution", "--base", "invalid_base_ref_999", "--target", "HEAD"])
    assert result.exit_code != 0
    assert "invalid_base_ref_999" in result.output


def test_symbol_change_types_added_deleted_modified(tmp_path):
    """
    Stage 1 Regression Test:
    In a controlled Git repo, test:
    - Added function -> ChangeType.ADDED
    - Deleted function -> ChangeType.DELETED
    - Modified function -> ChangeType.MODIFIED with exact line range and file association.
    """
    # 1. Initialize temporary git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@test.ai"], cwd=str(tmp_path), check=True, capture_output=True)

    service_file = tmp_path / "service.py"

    # Commit 1 (base): functions to be modified and deleted
    c1_content = (
        "def func_to_modify(x: int) -> int:\n"
        "    return x + 1\n\n"
        "def func_to_delete(y: str) -> str:\n"
        "    return y.strip()\n"
    )
    service_file.write_text(c1_content, encoding="utf-8")
    subprocess.run(["git", "add", "service.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=str(tmp_path), check=True, capture_output=True)

    c1_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    # Commit 2 (target): modify func_to_modify, delete func_to_delete, add func_to_add
    c2_content = (
        "def func_to_modify(x: int) -> int:\n"
        "    # Modified body\n"
        "    return (x * 2) + 1\n\n"
        "def func_to_add(z: float) -> float:\n"
        "    return z * 10.0\n"
    )
    service_file.write_text(c2_content, encoding="utf-8")
    subprocess.run(["git", "add", "service.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "commit 2"], cwd=str(tmp_path), check=True, capture_output=True)

    c2_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    diff_text, changed_files, _ = engine.resolve_git_diff(c1_sha, c2_sha)
    assert "service.py" in changed_files

    symbols = engine.extract_changed_symbols(
        diff_text=diff_text,
        changed_files=changed_files,
        base_ref=c1_sha,
        target_ref=c2_sha,
        renamed_files=engine.last_renamed_files,
        file_statuses=engine.last_file_statuses,
    )

    sym_dict = {s.name: s for s in symbols}
    assert "func_to_modify" in sym_dict
    assert "func_to_delete" in sym_dict
    assert "func_to_add" in sym_dict

    assert sym_dict["func_to_add"].change_type == ChangeType.ADDED
    assert sym_dict["func_to_delete"].change_type == ChangeType.DELETED
    assert sym_dict["func_to_modify"].change_type == ChangeType.MODIFIED
    assert sym_dict["func_to_modify"].file_path == "service.py"
    assert sym_dict["func_to_modify"].line_start == 1


def test_deleted_file_safe_git_object_retrieval(tmp_path):
    """
    Stage 1 Regression Test:
    When a file is deleted between revisions, its symbols must be safely retrieved
    from Git object storage at base_ref without relying on the file existing in the working tree.
    """
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@test.ai"], cwd=str(tmp_path), check=True, capture_output=True)

    helper_file = tmp_path / "deleted_helper.py"
    helper_file.write_text("def old_helper():\n    return 42\n", encoding="utf-8")
    subprocess.run(["git", "add", "deleted_helper.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add helper"], cwd=str(tmp_path), check=True, capture_output=True)
    c1_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    # Delete the file in commit 2
    subprocess.run(["git", "rm", "deleted_helper.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove helper"], cwd=str(tmp_path), check=True, capture_output=True)
    c2_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    # The file does NOT exist on disk in the working tree
    assert not helper_file.exists()

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    diff_text, changed_files, _ = engine.resolve_git_diff(c1_sha, c2_sha)
    assert "deleted_helper.py" in changed_files

    symbols = engine.extract_changed_symbols(
        diff_text=diff_text,
        changed_files=changed_files,
        base_ref=c1_sha,
        target_ref=c2_sha,
        renamed_files=engine.last_renamed_files,
        file_statuses=engine.last_file_statuses,
    )
    assert len(symbols) == 1
    assert symbols[0].name == "old_helper"
    assert symbols[0].change_type == ChangeType.DELETED
    assert symbols[0].file_path == "deleted_helper.py"


def test_renamed_file_preserves_paths(tmp_path):
    """
    Stage 1 Regression Test:
    When a file is renamed, Git status and rename headers preserve both old and new paths.
    """
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@test.ai"], cwd=str(tmp_path), check=True, capture_output=True)

    old_file = tmp_path / "old_name.py"
    old_file.write_text("def calculate_price(p: float) -> float:\n    return p * 1.1\n", encoding="utf-8")
    subprocess.run(["git", "add", "old_name.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "original"], cwd=str(tmp_path), check=True, capture_output=True)
    c1_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    # Git rename
    subprocess.run(["git", "mv", "old_name.py", "new_name.py"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "renamed"], cwd=str(tmp_path), check=True, capture_output=True)
    c2_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tmp_path), check=True, capture_output=True, text=True).stdout.strip()

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    diff_text, changed_files, _ = engine.resolve_git_diff(c1_sha, c2_sha)

    assert "new_name.py" in changed_files
    assert engine.last_renamed_files.get("new_name.py") == "old_name.py"


def test_evaluator_metrics_edge_cases():
    """
    Stage 2 Regression Test:
    Tests precision, recall, F1 edge cases:
    - True positives, false positives, false negatives
    - Both empty sets -> P=1.0, R=1.0, F1=1.0
    - Predicted non-empty but expected empty -> P=0.0, R=1.0, F1=0.0
    - Predicted empty but expected non-empty -> P=0.0, R=0.0, F1=0.0
    """
    from testpilot.evolution.evaluator import compute_prf

    # 1. Perfect match
    p, r, f1, tp, fp, fn = compute_prf({"a", "b"}, {"a", "b"})
    assert p == 1.0 and r == 1.0 and f1 == 1.0
    assert tp == ["a", "b"] and fp == [] and fn == []

    # 2. Both empty
    p, r, f1, tp, fp, fn = compute_prf(set(), set())
    assert p == 1.0 and r == 1.0 and f1 == 1.0
    assert tp == [] and fp == [] and fn == []

    # 3. Expected empty, predicted non-empty (False Positive only)
    p, r, f1, tp, fp, fn = compute_prf({"spurious"}, set())
    assert p == 0.0 and r == 1.0 and f1 == 0.0
    assert fp == ["spurious"] and fn == []

    # 4. Expected non-empty, predicted empty (False Negative only)
    p, r, f1, tp, fp, fn = compute_prf(set(), {"missed"})
    assert p == 0.0 and r == 0.0 and f1 == 0.0
    assert fn == ["missed"] and fp == []

    # 5. Mixed TP, FP, FN
    p, r, f1, tp, fp, fn = compute_prf({"a", "fp1"}, {"a", "fn1"})
    assert p == 0.5 and r == 0.5 and f1 == 0.5
    assert tp == ["a"] and fp == ["fp1"] and fn == ["fn1"]


def test_caller_resolution_six_controlled_fixtures(tmp_path):
    """
    Stage 4 Regression Tests:
    Validates the 6 required caller-resolution scenarios:
    1. Direct imported function call
    2. Aliased imported function call
    3. Two unrelated functions with same name in different modules (no false relationship)
    4. Unrelated object method with same name as changed function (e.g. worker.run())
    5. Test invoking a function through a helper
    6. Test invoking code through API route / fixture (with documented limitation)
    """
    from testpilot.sourcegraph.client import LocalCodeGraphFallback

    # Create module structure
    app_dir = tmp_path / "pkg"
    app_dir.mkdir(parents=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    # 1. Target module: pkg/payment.py
    (app_dir / "payment.py").write_text(
        "def process_charge(amount: float) -> bool:\n"
        "    return amount > 0\n\n"
        "def run():\n"
        "    pass\n",
        encoding="utf-8",
    )

    # 2. Unrelated module with same function name: pkg/unrelated.py
    (app_dir / "unrelated.py").write_text(
        "def process_charge(text: str) -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    # 3. Caller file: pkg/caller.py
    (app_dir / "caller.py").write_text(
        "from pkg.payment import process_charge\n"
        "from pkg.payment import process_charge as aliased_charge\n"
        "from pkg.unrelated import process_charge as unrelated_charge\n\n"
        "def direct_call_wrapper():\n"
        "    process_charge(100.0)\n\n"
        "def aliased_call_wrapper():\n"
        "    aliased_charge(200.0)\n\n"
        "def unrelated_call_wrapper():\n"
        "    unrelated_charge('test')\n\n"
        "def generic_method_caller(worker):\n"
        "    worker.run()\n",
        encoding="utf-8",
    )

    # 4. Test file with helper: tests/test_orders.py
    (tests_dir / "test_orders.py").write_text(
        "from pkg.payment import process_charge\n\n"
        "def helper_charge_user():\n"
        "    process_charge(50.0)\n\n"
        "def test_order_via_helper():\n"
        "    helper_charge_user()\n",
        encoding="utf-8",
    )

    fallback = LocalCodeGraphFallback(repo_root=str(tmp_path))

    # Fixture 1 & 2 & 3: Query callers of process_charge in pkg/payment.py
    callers = fallback.find_callers("process_charge", "pkg/payment.py")
    caller_names = {c["caller_name"] for c in callers}

    # Fixture 1: Direct imported call MUST be detected
    assert "direct_call_wrapper" in caller_names

    # Fixture 2: Aliased imported call MUST be detected
    assert "aliased_call_wrapper" in caller_names

    # Fixture 3: Unrelated call from different module MUST NOT be matched as caller of payment.py
    assert "unrelated_call_wrapper" not in caller_names

    # Fixture 4: Unrelated method worker.run() MUST NOT be treated as calling payment.py:run
    run_callers = fallback.find_callers("run", "pkg/payment.py")
    run_caller_names = {c["caller_name"] for c in run_callers}
    assert "generic_method_caller" not in run_caller_names

    # Fixture 5: Test invoking through helper
    # helper_charge_user calls process_charge directly
    assert "helper_charge_user" in caller_names
    # Now run engine test prioritization
    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(
        name="process_charge",
        file_path="pkg/payment.py",
        line_start=1,
        line_end=5,
        change_type=ChangeType.MODIFIED,
    )
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=3)
    prioritized = engine.prioritize_tests([sym], direct, indirect)
    p_names = {t.test_name for t in prioritized}
    # test_order_via_helper calls helper_charge_user (which is direct caller of process_charge)
    assert "test_order_via_helper" in p_names
    test_node = next(t for t in prioritized if t.test_name == "test_order_via_helper")
    assert test_node.priority_tier == PriorityTier.HIGH  # Tier 2: 1-hop via helper

    # Fixture 6: Verify route limitation behavior
    # Static AST matching correctly identifies direct/aliased/helper Python calls.
    # HTTP path strings (e.g. client.post("/pay")) are recorded as an architectural limitation.
    assert test_node.call_depth == 2


def test_evidence_backed_impact_explanations():
    """Verify ImpactNode and EvidenceTrail provide evidence-backed explanations and uncertainty reasons."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    sym = ChangedSymbol(
        name="calculate_discount",
        file_path="testbed/app/services/order_service.py",
        line_start=1,
        line_end=50,
        change_type=ChangeType.MODIFIED,
    )
    direct_nodes, indirect_nodes = engine.build_transitive_impact_graph([sym], max_depth=3)

    assert len(direct_nodes) > 0
    node = direct_nodes[0]

    # Verify initiating symbol & file
    assert node.initiating_symbol == "calculate_discount"
    assert node.initiating_file == "testbed/app/services/order_service.py"
    assert node.relationship_type == "direct"

    # Verify evidence trail
    assert node.evidence.evidence_source in ("Sourcegraph OSS", "Local AST Call Graph")
    assert len(node.evidence.call_chain) >= 2
    assert "calculate_discount" in node.evidence.call_path_description

    # Verify confidence & uncertainty
    assert 0.0 <= node.confidence <= 1.0
    assert 0.0 <= node.uncertainty_score <= 1.0
    assert len(node.uncertainty_reason) > 0
    assert isinstance(node.is_confirmed, bool)
    assert isinstance(node.limitations, list)


def test_explainable_test_prioritization():
    """Verify PrioritizedTest fields explain why each test was selected and its impact distance."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    sym = ChangedSymbol(
        name="calculate_discount",
        file_path="testbed/app/services/order_service.py",
        line_start=1,
        line_end=50,
        change_type=ChangeType.MODIFIED,
    )
    direct_nodes, indirect_nodes = engine.build_transitive_impact_graph([sym], max_depth=3)
    prioritized = engine.prioritize_tests([sym], direct_nodes, indirect_nodes)

    assert len(prioritized) > 0
    test_item = prioritized[0]

    # Explainability fields
    assert len(test_item.selection_reason) > 0
    assert len(test_item.targeted_symbol) > 0
    assert test_item.impact_distance in ("direct", "1 hop", "1 hop (helper)", "2 hops", "3 hops")
    assert test_item.evidence_type in ("confirmed", "heuristic")
    assert len(test_item.explanation) > 0
    assert test_item.execution_command.startswith("pytest ")


def test_evaluator_controlled_fixtures_suite():
    """Verify evaluator runs 6 controlled fixtures with 100% pass rate."""
    from testpilot.evolution.evaluator import EvolutionEvaluator

    evaluator = EvolutionEvaluator(repo_root=".")
    results = evaluator.evaluate_controlled_fixtures()

    assert "direct_caller" in results
    assert "multi_hop_caller" in results
    assert "same_name_isolation" in results
    assert "import_alias" in results
    assert "added_deleted_symbol" in results
    assert "dynamic_uncertainty" in results

    for name, res in results.items():
        assert res["passed"] is True, f"Controlled fixture {name} failed: {res}"
        assert res["precision"] == 1.0, f"{name} precision was {res['precision']}"
        assert res["recall"] == 1.0, f"{name} recall was {res['recall']}"
        assert res["f1_score"] == 1.0, f"{name} F1 was {res['f1_score']}"


def test_ast_extraction_calls_and_attributes(tmp_path):
    """
    Unit test verifying:
    - Plain function calls: function_name() is extracted
    - Attribute calls: self.method() is extracted as method
    - Module-qualified calls: module.function_name() is extracted as function_name
    - Local helper calls are recognized and expanded
    - Multiple calls in one test are all extracted
    - Unrelated common verbs (e.g. self.assertEqual, client.get) do NOT match changed symbols
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.models import ChangedSymbol, ChangeType, PriorityTier

    test_file_content = """
from pkg.service import calculate_tax

class OrderTests:
    def helper_setup_cart(self):
        calculate_tax(100.0)

    def test_complete_flow(self):
        self.helper_setup_cart()
        calculate_tax(50.0)
        self.assertEqual(1, 1)
        self.assertTrue(True)
        response = client.get("/api/v1/orders")
        close()
"""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_flow.py").write_text(test_file_content, encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))

    # Symbol 1: calculate_tax (should match test directly and via helper)
    sym_tax = ChangedSymbol(
        name="calculate_tax",
        file_path="pkg/service.py",
        line_start=1,
        line_end=10,
        change_type=ChangeType.MODIFIED,
    )
    # Symbol 2: unrelated common verb 'get' from unimported module (must NOT match client.get())
    sym_get = ChangedSymbol(
        name="get",
        file_path="pkg/other.py",
        line_start=1,
        line_end=10,
        change_type=ChangeType.MODIFIED,
    )
    # Symbol 3: unrelated assertEqual (must NOT match self.assertEqual())
    sym_assert = ChangedSymbol(
        name="assertEqual",
        file_path="pkg/testing.py",
        line_start=1,
        line_end=10,
        change_type=ChangeType.MODIFIED,
    )

    prioritized = engine.prioritize_tests([sym_tax, sym_get, sym_assert], [], [])

    # The test must be prioritized for calculate_tax, NOT for 'get' or 'assertEqual'
    assert len(prioritized) == 1
    t = prioritized[0]
    assert t.targeted_symbol == "calculate_tax"
    assert t.priority_tier == PriorityTier.CRITICAL
    assert t.targeted_symbol != "get"
    assert t.targeted_symbol != "assertEqual"
    assert "test_complete_flow" in t.test_name


def test_event_registration_framework_independent(tmp_path):
    """
    Verifies event registration detection on a framework-independent fixture:
    - Statically identifiable handler registration is detected
    - Relationship preserves source file, line number, and registration expression
    - Dynamic/unresolved receiver is marked uncertain
    - Unrelated .connect() calls (e.g. database / socket connections) are rejected
    - Event registration contributes to the impact graph with distinct ImpactType.EVENT_REGISTRATION
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.events import EventRegistrationDetector
    from testpilot.evolution.models import ChangedSymbol, ChangeType, ImpactType

    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)

    event_code = """
class Event:
    def connect(self, receiver):
        self.receivers.append(receiver)
    def send(self):
        for r in self.receivers:
            r()

order_placed = Event()

def changed_handler():
    pass

def dynamic_handler():
    pass

# Confirmed static registration (positional)
order_placed.connect(changed_handler)

# Confirmed static registration (keyword)
order_placed.connect(receiver=changed_handler)

# Dynamic / unresolved registration
order_placed.connect(getattr(dynamic_handler, 'resolve', None))

# Unrelated DB connection - MUST NOT be treated as event registration
class DB:
    def connect(self, host="localhost", port=5432):
        pass

db = DB()
db.connect(host="localhost", port=5432)
"""
    (app_dir / "events.py").write_text(event_code, encoding="utf-8")

    detector = EventRegistrationDetector(repo_root=str(tmp_path))

    # 1. Statically identifiable registration
    regs = detector.find_registrations_for_handler("changed_handler")
    assert len(regs) >= 2
    reg = regs[0]
    assert reg.event_name == "order_placed"
    assert reg.handler_name == "changed_handler"
    assert reg.is_confirmed is True
    assert reg.confidence == 0.85
    assert reg.uncertainty_score == 0.15
    assert "order_placed.connect" in reg.registration_expr
    assert reg.source_file == "app/events.py"
    assert reg.line_number > 0

    # 2. Dynamic registration check
    all_regs = detector.scan_file_for_registrations(app_dir / "events.py")
    dynamic_regs = [r for r in all_regs if not r.is_confirmed]
    assert len(dynamic_regs) >= 1
    d_reg = dynamic_regs[0]
    assert d_reg.is_confirmed is False
    assert d_reg.uncertainty_score == 0.50
    assert d_reg.confidence == 0.40

    # 3. Unrelated DB connection must NOT be in registrations
    assert not any("host=" in r.registration_expr for r in all_regs)
    assert not any(r.event_name == "db" for r in all_regs)

    # 4. Impact graph integration
    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(
        name="changed_handler",
        file_path="app/events.py",
        line_start=10,
        line_end=15,
        change_type=ChangeType.MODIFIED,
    )
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=3)

    event_nodes = [d for d in direct if d.impact_type == ImpactType.EVENT_REGISTRATION]
    assert len(event_nodes) >= 1
    enode = event_nodes[0]
    assert enode.symbol_name == "event:order_placed"
    assert enode.root_changed_symbol == "changed_handler"
    assert enode.is_confirmed is True
    assert enode.evidence.resolution_engine == "event_registration_detector"
    assert "order_placed" in enode.reason


def test_test_prioritization_event_and_deterministic_order(tmp_path):
    """
    Verifies test prioritization across:
    1. Direct call to changed symbol (Tier 1: CRITICAL)
    2. Helper call to changed symbol (Tier 1b: HIGH)
    3. Direct caller invocation (Tier 2: HIGH)
    4. Event trigger invocation (Tier 1c: HIGH)
    5. No duplicate tests and completely deterministic ordering
    6. Exclusion of setup/teardown methods from test recommendations
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.models import (
        ChangedSymbol,
        ChangeType,
        EvidenceTrail,
        ImpactNode,
        ImpactType,
        PriorityTier,
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    test_content = """
class SampleTestSuite:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def local_helper(self):
        target_fn()

    def test_direct(self):
        target_fn()

    def test_via_helper(self):
        self.local_helper()

    def test_event_trigger(self):
        event_obj.send()

    def test_caller(self):
        direct_caller_fn()
"""
    (tests_dir / "test_sample.py").write_text(test_content, encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))

    target_sym = ChangedSymbol(
        name="target_fn",
        file_path="app/core.py",
        line_start=1,
        line_end=10,
        change_type=ChangeType.MODIFIED,
    )
    direct_caller_node = ImpactNode(
        symbol_name="direct_caller_fn",
        file_path="app/service.py",
        line_number=15,
        depth=1,
        impact_type=ImpactType.DIRECT,
        root_changed_symbol="target_fn",
        root_changed_file="app/core.py",
        evidence=EvidenceTrail(
            call_chain=["target_fn", "direct_caller_fn"],
            caller_file="app/service.py",
            line_number=15,
        ),
        uncertainty_score=0.10,
        confidence=0.90,
        reason="Direct caller",
    )
    event_node = ImpactNode(
        symbol_name="event:event_obj",
        file_path="app/events.py",
        line_number=20,
        depth=1,
        impact_type=ImpactType.EVENT_REGISTRATION,
        root_changed_symbol="target_fn",
        root_changed_file="app/core.py",
        evidence=EvidenceTrail(
            call_chain=["target_fn", "event:event_obj"],
            caller_file="app/events.py",
            line_number=20,
            evidence_source="Event Registration",
        ),
        uncertainty_score=0.15,
        confidence=0.85,
        reason="Event registration",
    )

    prioritized = engine.prioritize_tests(
        changed_symbols=[target_sym],
        direct_impacts=[direct_caller_node, event_node],
        indirect_impacts=[],
    )

    # 1. No duplicates
    test_names = [t.test_name for t in prioritized]
    assert len(test_names) == len(set(test_names))

    # 2. Setup / teardown must NOT be treated as test cases
    assert not any("setUp" in name for name in test_names)
    assert not any("tearDown" in name for name in test_names)

    # 3. Direct test must have CRITICAL tier
    direct_t = next(t for t in prioritized if "test_direct" in t.test_name)
    assert direct_t.priority_tier == PriorityTier.CRITICAL
    assert direct_t.priority_score == 0.98

    # 4. Helper test must have HIGH tier
    helper_t = next(t for t in prioritized if "test_via_helper" in t.test_name)
    assert helper_t.priority_tier == PriorityTier.HIGH
    assert helper_t.priority_score == 0.85

    # 5. Event trigger test without behavioral evidence is classified as EXPLICIT_EVENT_DISPATCH (Tier MEDIUM, score 0.70)
    event_t = next(t for t in prioritized if "test_event_trigger" in t.test_name)
    assert event_t.priority_tier == PriorityTier.MEDIUM
    assert event_t.priority_score == 0.70
    assert event_t.match_classification.value == "EXPLICIT_EVENT_DISPATCH"
    assert "event_obj" in event_t.impact_distance
    assert "target_fn" in event_t.explanation

    # 6. Caller test must have HIGH tier
    caller_t = next(t for t in prioritized if "test_caller" in t.test_name)
    assert caller_t.priority_tier == PriorityTier.HIGH
    assert caller_t.priority_score == 0.82

    # 7. Output ordering must be deterministic and sorted by score descending:
    # Direct (0.98) > Helper (0.85) > Caller (0.82) > Explicit Event (0.70)
    scores = [t.priority_score for t in prioritized]
    assert scores == sorted(scores, reverse=True)


def test_event_trigger_call_shape_heuristic_and_unverified_dispatch(tmp_path):
    """
    Regression test for call-shape-aware event trigger matching:
    1. call_command('migrate') produces an EVENT_TRIGGER_CANDIDATE with score 0.45 (MEDIUM),
       labeled evidence_type='heuristic', with explanation explicitly stating
       that behavior-specific coverage is unconfirmed.
    2. An unrelated function call with 'migrate' argument does NOT match post_migrate.
    3. A generic migrate() call does NOT produce a confirmed post_migrate event match.
    4. Explicit post_migrate.send(...) is classified as EXPLICIT_EVENT_DISPATCH (score 0.70, MEDIUM).
    5. Priority ordering is deterministic: direct changed (0.98) > explicit event (0.70) > candidate command (0.45).
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.models import (
        ChangedSymbol,
        ChangeType,
        EventMatchClassification,
        EvidenceTrail,
        ImpactNode,
        ImpactType,
        PriorityTier,
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    test_content = """
def test_direct_changed():
    changed_service_fn()

def test_explicit_event_dispatch():
    post_migrate.send(sender="test_app")

def test_command_dispatch_heuristic():
    call_command("migrate", verbosity=0)

def test_unrelated_call_with_migrate_string():
    logger.info("migrate")
    custom_func("migrate")

def test_generic_migrate_call():
    migrate()
"""
    (tests_dir / "test_migration_suite.py").write_text(test_content, encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))

    changed_sym = ChangedSymbol(
        name="changed_service_fn",
        file_path="app/core.py",
        line_start=10,
        line_end=25,
        change_type=ChangeType.MODIFIED,
    )

    event_node = ImpactNode(
        symbol_name="event:post_migrate",
        file_path="app/signals.py",
        line_number=5,
        depth=1,
        impact_type=ImpactType.EVENT_REGISTRATION,
        root_changed_symbol="handle_post_migrate",
        root_changed_file="app/handlers.py",
        evidence=EvidenceTrail(
            call_chain=["handle_post_migrate", "event:post_migrate"],
            caller_file="app/signals.py",
            line_number=5,
            evidence_source="Event Registration",
        ),
        uncertainty_score=0.15,
        confidence=0.85,
        is_confirmed=True,
        reason="Confirmed event registration for post_migrate",
    )

    prioritized = engine.prioritize_tests(
        changed_symbols=[changed_sym],
        direct_impacts=[event_node],
        indirect_impacts=[],
    )

    test_map = {t.test_name: t for t in prioritized}

    # 1. call_command("migrate") produces a candidate match, NOT automatically High
    assert "test_command_dispatch_heuristic" in test_map
    heuristic_t = test_map["test_command_dispatch_heuristic"]
    assert heuristic_t.evidence_type == "heuristic"
    assert heuristic_t.priority_score == 0.45
    assert heuristic_t.priority_tier == PriorityTier.MEDIUM
    assert heuristic_t.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert heuristic_t.event_name == "post_migrate"
    assert len(heuristic_t.missing_evidence) > 0
    assert "unconfirmed behavioral coverage" in heuristic_t.selection_reason.lower()

    # 2. Unrelated function call with "migrate" does NOT match post_migrate
    assert "test_unrelated_call_with_migrate_string" not in test_map

    # 3. Generic migrate() call does NOT produce a post_migrate event match
    assert "test_generic_migrate_call" not in test_map

    # 4. Explicit post_migrate.send(...) is classified as EXPLICIT_EVENT_DISPATCH (score 0.70)
    assert "test_explicit_event_dispatch" in test_map
    explicit_t = test_map["test_explicit_event_dispatch"]
    assert explicit_t.evidence_type == "confirmed"
    assert explicit_t.priority_score == 0.70
    assert explicit_t.priority_tier == PriorityTier.MEDIUM
    assert explicit_t.match_classification == EventMatchClassification.EXPLICIT_EVENT_DISPATCH
    assert "post_migrate" in explicit_t.explanation

    # 5. Direct changed-symbol match has top priority
    assert "test_direct_changed" in test_map
    direct_t = test_map["test_direct_changed"]
    assert direct_t.priority_score == 0.98
    assert direct_t.priority_tier == PriorityTier.CRITICAL
    assert direct_t.evidence_type == "confirmed"

    # 6. Priority ordering: direct (0.98) > explicit event (0.70) > heuristic candidate (0.45)
    assert direct_t.priority_score > explicit_t.priority_score > heuristic_t.priority_score
    prioritized_names = [t.test_name for t in prioritized]
    assert prioritized_names == [
        "test_direct_changed",
        "test_explicit_event_dispatch",
        "test_command_dispatch_heuristic",
    ]


def test_command_dispatch_record_extraction():
    """
    Verifies Section 6 requirements for _extract_command_dispatch_records:
    - call_command("migrate")
    - call_command(command_name="migrate")
    - self.call_command("migrate")
    - MigrationExecutor(...)
    - executor.migrate(...)
    - Unrelated string literals containing "migrate"
    - Unrecognized functions that receive "migrate"
    """
    import ast

    from testpilot.evolution.engine import RepositoryEvolutionEngine

    snippet = """
def test_various_dispatches(self):
    call_command("migrate", verbosity=0)
    call_command(command_name="migrate", interactive=False)
    self.call_command("migrate", "auth_tests")
    executor = MigrationExecutor(connection)
    executor.migrate([("auth", "0001_initial")])
    unrelated_fn("migrate")
    logger.info("migrate running")
    s = "we should migrate this"
    some_obj.do_something("migrate")
"""
    tree = ast.parse(snippet)
    records = RepositoryEvolutionEngine._extract_command_dispatch_records(tree)
    cmd_names = [r.command_name for r in records]
    apis = [r.dispatch_api for r in records]

    # Verify recognized dispatches
    assert "migrate" in cmd_names
    assert "call_command" in apis
    assert "self.call_command" in apis
    assert "MigrationExecutor" in apis
    assert "executor.migrate" in apis

    # Verify keywords and args captured
    kw_records = [r for r in records if r.dispatch_api == "call_command(keyword)"]
    assert len(kw_records) >= 1
    assert kw_records[0].command_name == "migrate"

    # Verify extra args captured for targeted migration
    self_records = [r for r in records if r.dispatch_api == "self.call_command"]
    assert any("auth_tests" in r.extra_args for r in self_records)

    # Verify unrelated function calls and strings are NOT captured
    assert "logger.info" not in apis
    assert "unrelated_fn" not in apis
    assert "some_obj.do_something" not in apis

    # Also test the backward-compatible _extract_command_dispatch_args
    args_set = RepositoryEvolutionEngine._extract_command_dispatch_args(tree)
    assert "migrate" in args_set
    assert "logger.info" not in args_set


def test_event_matching_matrix_all_scenarios(tmp_path):
    """
    Verifies Section 8 scenario matrix:
    - Explicit event dispatch -> EXPLICIT_EVENT_DISPATCH
    - Only call_command("migrate") -> EVENT_TRIGGER_CANDIDATE, not automatically High
    - Only MigrationExecutor(...) -> EVENT_TRIGGER_CANDIDATE, not automatically High
    - Only executor.migrate() -> EVENT_TRIGGER_CANDIDATE, not automatically High
    - Migration + behavioral evidence -> BEHAVIORAL_COVERAGE, High priority
    - Explicit event + behavioral evidence -> Both evidence sources, High priority
    - Unrelated "migrate" string -> No match
    - Unrelated command -> No post_migrate match
    - Local helper propagation -> Call path preserved
    - Direct-impact test -> Intact (CRITICAL)
    - Indirect-impact test -> Intact (MEDIUM)
    - Report separates prioritized regression tests from event_trigger_candidates
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.models import (
        ChangedSymbol,
        ChangeType,
        EventMatchClassification,
        EvidenceTrail,
        ImpactNode,
        ImpactType,
        PriorityTier,
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    test_content = """
class ComprehensiveEventTestSuite:
    def helper_running_migration(self):
        call_command("migrate")

    def test_explicit_dispatch_only(self):
        post_migrate.send(sender="test_app")

    def test_call_command_only(self):
        call_command("migrate", verbosity=0)

    def test_migration_executor_only(self):
        executor = MigrationExecutor(connection)

    def test_executor_migrate_only(self):
        self.executor.migrate([("auth", "0002_update")])

    def test_migration_with_behavioral_coverage(self):
        call_command("migrate", "auth_tests")
        perms = Permission.objects.filter(codename="test_perm")
        assert perms.exists()

    def test_explicit_event_with_behavioral_coverage(self):
        post_migrate.send(sender="auth_tests")
        perms = Permission.objects.filter(codename="test_perm")
        assert perms.count() == 1

    def test_unrelated_migrate_string(self):
        log_message("migrate process finished")
        custom_action("migrate")

    def test_unrelated_command(self):
        call_command("dumpdata", "auth")

    def test_via_helper_migration(self):
        self.helper_running_migration()

    def test_direct_changed(self):
        rename_permissions_after_model_rename()

    def test_indirect_changed(self):
        caller_of_rename()
"""
    (tests_dir / "test_matrix.py").write_text(test_content, encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))

    changed_sym = ChangedSymbol(
        name="rename_permissions_after_model_rename",
        file_path="django/contrib/auth/management/__init__.py",
        line_start=10,
        line_end=50,
        change_type=ChangeType.MODIFIED,
    )

    indirect_node = ImpactNode(
        symbol_name="caller_of_rename",
        file_path="django/contrib/auth/management/__init__.py",
        line_number=60,
        depth=2,
        impact_type=ImpactType.INDIRECT,
        root_changed_symbol="rename_permissions_after_model_rename",
        evidence=EvidenceTrail(
            call_chain=["rename_permissions_after_model_rename", "intermediate_fn", "caller_of_rename"],
            caller_file="django/contrib/auth/management/__init__.py",
            line_number=60,
        ),
        uncertainty_score=0.30,
        confidence=0.70,
        reason="Indirect call 2 hops",
    )

    event_node = ImpactNode(
        symbol_name="event:post_migrate",
        file_path="django/contrib/auth/apps.py",
        line_number=15,
        depth=1,
        impact_type=ImpactType.EVENT_REGISTRATION,
        root_changed_symbol="rename_permissions_after_model_rename",
        root_changed_file="django/contrib/auth/management/__init__.py",
        evidence=EvidenceTrail(
            call_chain=["rename_permissions_after_model_rename", "event:post_migrate"],
            caller_file="django/contrib/auth/apps.py",
            line_number=15,
            evidence_source="Event Registration",
        ),
        uncertainty_score=0.15,
        confidence=0.85,
        is_confirmed=True,
        reason="Confirmed post_migrate listener",
    )

    prioritized = engine.prioritize_tests(
        changed_symbols=[changed_sym],
        direct_impacts=[event_node],
        indirect_impacts=[indirect_node],
    )

    test_map = {t.test_name.split("::")[-1]: t for t in prioritized}

    # 1. Explicit event dispatch only -> EXPLICIT_EVENT_DISPATCH (score 0.70, MEDIUM)
    exp_test = test_map["test_explicit_dispatch_only"]
    assert exp_test.match_classification == EventMatchClassification.EXPLICIT_EVENT_DISPATCH
    assert exp_test.priority_tier == PriorityTier.MEDIUM
    assert exp_test.priority_score == 0.70

    # 2. call_command("migrate") only -> EVENT_TRIGGER_CANDIDATE (score 0.45, MEDIUM)
    cc_test = test_map["test_call_command_only"]
    assert cc_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert cc_test.priority_tier == PriorityTier.MEDIUM
    assert cc_test.priority_score == 0.45

    # 3. MigrationExecutor only -> EVENT_TRIGGER_CANDIDATE (score 0.45, MEDIUM)
    me_test = test_map["test_migration_executor_only"]
    assert me_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert me_test.priority_tier == PriorityTier.MEDIUM

    # 4. executor.migrate() only -> EVENT_TRIGGER_CANDIDATE (score 0.45, MEDIUM)
    em_test = test_map["test_executor_migrate_only"]
    assert em_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert em_test.priority_tier == PriorityTier.MEDIUM

    # 5. Migration + Behavioral evidence -> BEHAVIORAL_COVERAGE (score 0.85, HIGH)
    beh_test = test_map["test_migration_with_behavioral_coverage"]
    assert beh_test.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE
    assert beh_test.priority_tier == PriorityTier.HIGH
    assert beh_test.priority_score == 0.85
    assert any("Permission" in e.description for e in beh_test.structured_evidence)

    # 6. Explicit event + Behavioral evidence -> BEHAVIORAL_COVERAGE (score 0.88, HIGH)
    exp_beh_test = test_map["test_explicit_event_with_behavioral_coverage"]
    assert exp_beh_test.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE
    assert exp_beh_test.priority_tier == PriorityTier.HIGH
    assert exp_beh_test.priority_score == 0.88
    # Both explicit event and behavioral evidence are represented
    ev_types = [e.type for e in exp_beh_test.structured_evidence]
    assert "EXPLICIT_DISPATCH" in ev_types
    assert "BEHAVIORAL_ASSERTION" in ev_types

    # 7. Unrelated "migrate" string -> NOT matched
    assert "test_unrelated_migrate_string" not in test_map

    # 8. Unrelated command -> NOT matched
    assert "test_unrelated_command" not in test_map

    # 9. Local helper migration -> EVENT_TRIGGER_CANDIDATE with helper call chain
    helper_test = test_map["test_via_helper_migration"]
    assert helper_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert "helper_running_migration" in helper_test.evidence.call_path_description

    # 10. Direct changed symbol -> Intact CRITICAL (0.98)
    direct_test = test_map["test_direct_changed"]
    assert direct_test.priority_tier == PriorityTier.CRITICAL
    assert direct_test.priority_score == 0.98

    # 11. Indirect caller -> Intact MEDIUM (0.60)
    ind_test = test_map["test_indirect_changed"]
    assert ind_test.priority_tier == PriorityTier.MEDIUM
    assert ind_test.priority_score == 0.60

    # 12. Verify EvolutionReport partitioning
    # When report is created, prioritized_tests contains regression tests,
    # and event_trigger_candidates contains candidate tests with unconfirmed coverage.
    from testpilot.evolution.models import EvolutionReport
    reg_tests = [t for t in prioritized if t.match_classification != EventMatchClassification.EVENT_TRIGGER_CANDIDATE]
    cand_tests = [t for t in prioritized if t.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE]

    report = EvolutionReport(
        base_ref="HEAD~1",
        target_ref="HEAD",
        diff_stat="1 file changed",
        changed_files=["django/contrib/auth/management/__init__.py"],
        changed_symbols=[changed_sym],
        direct_impacts=[event_node],
        indirect_impacts=[indirect_node],
        total_impacted_symbols=2,
        prioritized_tests=reg_tests,
        event_trigger_candidates=cand_tests,
        total_discovered_tests=len(prioritized),
        analysis_latency_ms=5.0,
    )
    assert len(report.prioritized_tests) == 5  # direct, exp+beh, beh, exp, indirect (5 total regression)
    assert len(report.event_trigger_candidates) == 4  # call_command, MigrationExecutor, executor.migrate, via_helper
    assert report.total_discovered_tests == 9


def test_api_evolution_endpoint_serializes_candidates_and_evidence():
    """Verify that /api/evolution/analyze serializes event classifications and candidates."""
    from testpilot.evolution.models import (
        EventMatchClassification,
        EvidenceItem,
        EvolutionReport,
        PrioritizedTest,
        PriorityTier,
    )

    client = TestClient(app)
    refs_res = client.get("/api/evolution/refs")
    assert refs_res.status_code == 200

    # Model serialization check directly
    test_item = PrioritizedTest(
        test_name="test_perm_rename",
        test_file="tests/test_auth.py",
        line_number=20,
        priority_tier=PriorityTier.HIGH,
        priority_score=0.85,
        reason="Behavioral coverage established",
        targeted_symbol="rename_permissions",
        match_classification=EventMatchClassification.BEHAVIORAL_COVERAGE,
        event_name="post_migrate",
        structured_evidence=[
            EvidenceItem(
                type="BEHAVIORAL_ASSERTION",
                description="The test queries Permission models",
                source_file="tests/test_auth.py",
                line=25,
            )
        ],
        missing_evidence=[],
        selection_reason="Test verifies permission renaming behavior during migration",
        confidence=0.85,
    )

    cand_item = PrioritizedTest(
        test_name="test_generic_migrate",
        test_file="tests/test_generic.py",
        line_number=10,
        priority_tier=PriorityTier.MEDIUM,
        priority_score=0.45,
        reason="Trigger candidate",
        targeted_symbol="post_migrate",
        match_classification=EventMatchClassification.EVENT_TRIGGER_CANDIDATE,
        event_name="post_migrate",
        structured_evidence=[
            EvidenceItem(
                type="COMMAND_DISPATCH",
                description="The test invokes call_command('migrate')",
                source_file="tests/test_generic.py",
                line=12,
            )
        ],
        missing_evidence=["No behavior-specific assertion or changed-symbol relationship was established."],
        selection_reason="Test may trigger post_migrate through migration execution, but behavioral coverage is unconfirmed",
        confidence=0.45,
    )

    report = EvolutionReport(
        base_ref="HEAD~1",
        target_ref="HEAD",
        diff_stat="1 file changed",
        changed_files=["app.py"],
        changed_symbols=[],
        direct_impacts=[],
        indirect_impacts=[],
        total_impacted_symbols=0,
        prioritized_tests=[test_item],
        event_trigger_candidates=[cand_item],
        total_discovered_tests=2,
        analysis_latency_ms=10.0,
    )

    dumped = report.model_dump()
    assert "event_trigger_candidates" in dumped
    assert len(dumped["event_trigger_candidates"]) == 1
    assert dumped["event_trigger_candidates"][0]["match_classification"] == "EVENT_TRIGGER_CANDIDATE"
    assert dumped["prioritized_tests"][0]["match_classification"] == "BEHAVIORAL_COVERAGE"
    assert dumped["prioritized_tests"][0]["structured_evidence"][0]["type"] == "BEHAVIORAL_ASSERTION"
    assert dumped["total_discovered_tests"] == 2


def test_evolution_false_positive_elimination_and_token_boundaries(tmp_path):
    """
    Regression tests verifying:
    1. 'auth' does not match 'author_app', but matches 'auth_tests' and 'django.contrib.auth'.
    2. 'rename' does not match 'renamedfoo'.
    3. A migration command raising CLI argument errors is classified as CANDIDATE, not BEHAVIORAL_COVERAGE.
    4. Generic Permission.objects.all() usage is not treated as permission-renaming behavioral coverage.
    5. Genuine permission codename transition assertions remain HIGH / BEHAVIORAL_COVERAGE.
    """
    from testpilot.evolution.engine import RepositoryEvolutionEngine
    from testpilot.evolution.models import (
        ChangedSymbol,
        ChangeType,
        EventMatchClassification,
        EvidenceTrail,
        ImpactNode,
        ImpactType,
        PriorityTier,
    )

    # 1. Token boundary matching checks
    assert not RepositoryEvolutionEngine._matches_any_token("author_app", {"auth"})
    assert RepositoryEvolutionEngine._matches_any_token("auth_tests", {"auth"})
    assert RepositoryEvolutionEngine._matches_any_token("django.contrib.auth", {"auth"})

    # 2. Generic verb boundary checks
    assert not RepositoryEvolutionEngine._matches_any_token("renamedfoo", {"rename"})

    # Set up synthetic tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_src = """
class CliErrorTests:
    def test_cli_argument_error(self):
        with self.assertRaisesMessage(CommandError, "Did you mean auth?"):
            call_command("migrate", "django.contrib.auth")

class SwappablePermissionTests:
    def test_generic_permission_creation(self):
        call_command("migrate", interactive=False, verbosity=0)
        apps_models = [(p.content_type.app_label, p.content_type.model) for p in Permission.objects.all()]
        self.assertIn(("swappable_models", "alternatearticle"), apps_models)

class ContentTypeOnlyTests:
    def test_content_type_only_rename(self):
        call_command("migrate", "contenttypes_tests", verbosity=0)
        ContentType.objects.filter(app_label="contenttypes_tests", model="renamedfoo").exists()

class QueryWithoutAssertionTests:
    def test_query_without_assert(self):
        call_command("migrate", "auth_tests")
        perms = Permission.objects.filter(codename="change_newmodel")

class GenericKwargTests:
    def test_generic_kwargs_filter(self):
        call_command("migrate", "auth_tests")
        perms = Permission.objects.filter(using="other")
        assert perms.exists()

class AuthorAppTests:
    def test_author_app_migrate(self):
        call_command("migrate", "author_app")

class GenuinePermissionRenameTests:
    def test_genuine_codename_transition(self):
        call_command("migrate", "auth_tests", "0002", verbosity=0)
        perms = Permission.objects.filter(codename="change_newmodel")
        assert perms.exists()

    def test_genuine_verbosity_output(self):
        call_command("migrate", "auth_tests", "0002", verbosity=2, stdout=self.stdout)
        self.assertIn("Renamed permission(s): auth_tests.add_oldmodel → add_newmodel", self.stdout.getvalue())
"""
    (tests_dir / "test_regressions.py").write_text(test_src, encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))

    changed_sym = ChangedSymbol(
        name="rename_permissions_after_model_rename",
        file_path="django/contrib/auth/management/__init__.py",
        line_start=10,
        line_end=50,
        change_type=ChangeType.MODIFIED,
    )
    event_node = ImpactNode(
        symbol_name="event:post_migrate",
        file_path="django/contrib/auth/apps.py",
        line_number=15,
        depth=1,
        impact_type=ImpactType.EVENT_REGISTRATION,
        root_changed_symbol="rename_permissions_after_model_rename",
        root_changed_file="django/contrib/auth/management/__init__.py",
        evidence=EvidenceTrail(
            call_chain=["rename_permissions_after_model_rename", "event:post_migrate"],
            caller_file="django/contrib/auth/apps.py",
            line_number=15,
            evidence_source="Event Registration",
        ),
        uncertainty_score=0.15,
        confidence=0.85,
        is_confirmed=True,
        reason="Confirmed post_migrate listener",
    )

    prioritized = engine.prioritize_tests(
        changed_symbols=[changed_sym],
        direct_impacts=[event_node],
        indirect_impacts=[],
        include_candidates=True,
    )

    test_map = {t.test_name.split("::")[-1]: t for t in prioritized}

    # 3. CLI argument error test -> Candidate (not behavioral coverage)
    cli_test = test_map["test_cli_argument_error"]
    assert cli_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert cli_test.priority_tier == PriorityTier.MEDIUM
    assert cli_test.priority_score == 0.45

    # 4. Generic Permission.objects.all() -> Candidate (not behavioral coverage)
    gen_test = test_map["test_generic_permission_creation"]
    assert gen_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert gen_test.priority_tier == PriorityTier.MEDIUM
    assert gen_test.priority_score == 0.45

    # 4b. ContentType only test -> Candidate (not behavioral coverage)
    ct_test = test_map["test_content_type_only_rename"]
    assert ct_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert ct_test.priority_tier == PriorityTier.MEDIUM

    # 4c. Query without assertion -> Candidate (a query is not an assertion)
    q_no_assert = test_map["test_query_without_assert"]
    assert q_no_assert.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert q_no_assert.priority_tier == PriorityTier.MEDIUM
    assert not any(e.type == "BEHAVIORAL_ASSERTION" for e in q_no_assert.structured_evidence)

    # 4d. Generic kwargs on filter -> Candidate (must have relevant fields like codename or name)
    gen_kw_test = test_map["test_generic_kwargs_filter"]
    assert gen_kw_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    assert gen_kw_test.priority_tier == PriorityTier.MEDIUM
    assert not any(e.type == "BEHAVIORAL_ASSERTION" for e in gen_kw_test.structured_evidence)

    # 4e. Author app -> Candidate without auth targeted argument dispatch
    author_test = test_map["test_author_app_migrate"]
    assert author_test.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
    # Verify 'author_app' was not matched as an 'auth' domain argument
    assert not any("targeted arguments" in e.description for e in author_test.structured_evidence)

    # 5. Genuine codename transition -> HIGH / BEHAVIORAL_COVERAGE
    genuine_codename = test_map["test_genuine_codename_transition"]
    assert genuine_codename.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE
    assert genuine_codename.priority_tier == PriorityTier.HIGH
    assert genuine_codename.priority_score == 0.85
    assert any(
        "Permission.objects.filter" in e.description and "codename" in e.description
        for e in genuine_codename.structured_evidence
    )

    # 5b. Genuine stdout verbosity output assertion -> HIGH / BEHAVIORAL_COVERAGE
    genuine_stdout = test_map["test_genuine_verbosity_output"]
    assert genuine_stdout.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE
    assert genuine_stdout.priority_tier == PriorityTier.HIGH
    assert genuine_stdout.priority_score == 0.85


# =========================================================================
# Targeted Regression Tests: Transitive Call-Graph Blast Radius
# =========================================================================

def test_transitive_blast_radius_direct_caller_detection():
    """1. Direct caller detection: calculate_tax is called by calculate_order_totals."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    sym = ChangedSymbol(
        name="calculate_tax",
        class_name="OrderService",
        file_path="testbed/app/services/order_service.py",
        line_start=63,
        line_end=75,
        change_type=ChangeType.MODIFIED,
    )
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=1)
    direct_names = [d.symbol_name for d in direct if d.impact_type == ImpactType.DIRECT]
    assert "calculate_order_totals" in direct_names
    assert len(indirect) == 0
    node = next(d for d in direct if d.symbol_name == "calculate_order_totals")
    assert node.depth == 1
    assert node.impact_type == ImpactType.DIRECT
    assert node.evidence.evidence_source in ("Local AST Call Graph", "Sourcegraph OSS")


def test_transitive_blast_radius_one_level_transitive_caller():
    """2. One-level transitive caller: calculate_tax -> calculate_order_totals -> create_order (depth 2)."""
    engine = RepositoryEvolutionEngine(repo_root=".")
    sym = ChangedSymbol(
        name="calculate_tax",
        class_name="OrderService",
        file_path="testbed/app/services/order_service.py",
        line_start=63,
        line_end=75,
        change_type=ChangeType.MODIFIED,
    )
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=2)
    direct_names = [d.symbol_name for d in direct if d.impact_type == ImpactType.DIRECT]
    indirect_names = [i.symbol_name for i in indirect if i.impact_type == ImpactType.INDIRECT]

    assert "calculate_order_totals" in direct_names
    assert "create_order" in indirect_names
    ind_node = next(i for i in indirect if i.symbol_name == "create_order")
    assert ind_node.depth == 2
    assert ind_node.impact_type == ImpactType.INDIRECT
    assert "calculate_order_totals" in ind_node.evidence.call_chain


def test_transitive_blast_radius_multi_level_transitive_fixture(tmp_path):
    """3. Multi-level transitive caller: A -> B -> C -> D across files and depth hops."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text(
        "def func_a():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "b.py").write_text(
        "from pkg.a import func_a\n"
        "def func_b():\n"
        "    return func_a() + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "c.py").write_text(
        "from pkg.b import func_b\n"
        "def func_c():\n"
        "    return func_b() * 2\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "d.py").write_text(
        "from pkg.c import func_c\n"
        "def func_d():\n"
        "    return func_c() - 5\n",
        encoding="utf-8",
    )

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(
        name="func_a",
        file_path="pkg/a.py",
        line_start=1,
        line_end=3,
        change_type=ChangeType.MODIFIED,
    )
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=3)
    d_names = [d.symbol_name for d in direct]
    i_names = [i.symbol_name for i in indirect]

    assert "func_b" in d_names  # Depth 1
    assert "func_c" in i_names  # Depth 2
    assert "func_d" in i_names  # Depth 3

    node_d = next(i for i in indirect if i.symbol_name == "func_d")
    assert node_d.depth == 3
    assert node_d.evidence.call_chain == ["func_a", "func_b", "func_c", "func_d"]


def test_transitive_blast_radius_max_depth_enforcement(tmp_path):
    """4. max_depth parameter strictly bounds the traversal depth."""
    (tmp_path / "mod").mkdir()
    (tmp_path / "mod" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "mod" / "step0.py").write_text("def step_0(): return 0\n", encoding="utf-8")
    (tmp_path / "mod" / "step1.py").write_text("from mod.step0 import step_0\ndef step_1(): return step_0()\n", encoding="utf-8")
    (tmp_path / "mod" / "step2.py").write_text("from mod.step1 import step_1\ndef step_2(): return step_1()\n", encoding="utf-8")
    (tmp_path / "mod" / "step3.py").write_text("from mod.step2 import step_2\ndef step_3(): return step_2()\n", encoding="utf-8")

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(name="step_0", file_path="mod/step0.py", line_start=1, line_end=2)

    # Depth 1: Only step_1
    d1, ind1 = engine.build_transitive_impact_graph([sym], max_depth=1)
    assert [d.symbol_name for d in d1] == ["step_1"]
    assert len(ind1) == 0

    # Depth 2: step_1 (direct) + step_2 (indirect)
    d2, ind2 = engine.build_transitive_impact_graph([sym], max_depth=2)
    assert [d.symbol_name for d in d2] == ["step_1"]
    assert [i.symbol_name for i in ind2] == ["step_2"]

    # Depth 3: step_1 (direct) + step_2, step_3 (indirect)
    d3, ind3 = engine.build_transitive_impact_graph([sym], max_depth=3)
    assert [d.symbol_name for d in d3] == ["step_1"]
    assert {i.symbol_name for i in ind3} == {"step_2", "step_3"}


def test_transitive_blast_radius_cycle_prevention(tmp_path):
    """5. Cycle prevention: Mutually recursive functions (ping -> pong -> ping) terminate cleanly."""
    (tmp_path / "cyc").mkdir()
    (tmp_path / "cyc" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "cyc" / "loop.py").write_text(
        "def ping(n):\n"
        "    if n > 0:\n"
        "        return pong(n - 1)\n"
        "    return 0\n\n"
        "def pong(n):\n"
        "    if n > 0:\n"
        "        return ping(n - 1)\n"
        "    return 0\n",
        encoding="utf-8",
    )

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(name="ping", file_path="cyc/loop.py", line_start=1, line_end=5)
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=4)

    # pong calls ping directly -> depth 1
    assert len(direct) == 1
    assert direct[0].symbol_name == "pong"
    # ping itself must NOT be re-added as an indirect caller of itself!
    assert not any(i.symbol_name == "ping" for i in indirect)


def test_transitive_blast_radius_duplicate_prevention(tmp_path):
    """6. Duplicate prevention: Diamond dependencies (A -> B -> D, A -> C -> D) only record D once."""
    (tmp_path / "diamond").mkdir()
    (tmp_path / "diamond" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "diamond" / "base.py").write_text("def base_fn(): return 1\n", encoding="utf-8")
    (tmp_path / "diamond" / "left.py").write_text(
        "from diamond.base import base_fn\n"
        "def left_fn(): return base_fn()\n",
        encoding="utf-8",
    )
    (tmp_path / "diamond" / "right.py").write_text(
        "from diamond.base import base_fn\n"
        "def right_fn(): return base_fn()\n",
        encoding="utf-8",
    )
    (tmp_path / "diamond" / "top.py").write_text(
        "from diamond.left import left_fn\n"
        "from diamond.right import right_fn\n"
        "def top_fn(): return left_fn() + right_fn()\n",
        encoding="utf-8",
    )

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(name="base_fn", file_path="diamond/base.py", line_start=1, line_end=2)
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=3)

    assert {d.symbol_name for d in direct} == {"left_fn", "right_fn"}
    # top_fn should appear exactly once in indirect impacts
    top_matches = [i for i in indirect if i.symbol_name == "top_fn"]
    assert len(top_matches) == 1


def test_transitive_blast_radius_qualified_symbol_collision_avoidance(tmp_path):
    """7. Qualified symbol collision: Two classes sharing the method name 'execute' are distinguished."""
    (tmp_path / "workers").mkdir()
    (tmp_path / "workers" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "workers" / "payment_worker.py").write_text(
        "class PaymentWorker:\n"
        "    def execute(self):\n"
        "        return 'paid'\n",
        encoding="utf-8",
    )
    (tmp_path / "workers" / "email_worker.py").write_text(
        "class EmailWorker:\n"
        "    def execute(self):\n"
        "        return 'emailed'\n",
        encoding="utf-8",
    )
    (tmp_path / "workers" / "caller.py").write_text(
        "from workers.payment_worker import PaymentWorker\n"
        "from workers.email_worker import EmailWorker\n\n"
        "def run_payment():\n"
        "    w = PaymentWorker()\n"
        "    return w.execute()\n\n"
        "def run_email():\n"
        "    e = EmailWorker()\n"
        "    return e.execute()\n",
        encoding="utf-8",
    )

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    pay_sym = ChangedSymbol(
        name="execute",
        class_name="PaymentWorker",
        file_path="workers/payment_worker.py",
        line_start=2,
        line_end=4,
    )
    direct, indirect = engine.build_transitive_impact_graph([pay_sym], max_depth=2)

    # Caller of PaymentWorker.execute must be run_payment, NOT run_email
    d_names = [d.symbol_name for d in direct]
    assert "run_payment" in d_names
    assert "run_email" not in d_names


def test_transitive_blast_radius_zero_downstream_callers(tmp_path):
    """8. Zero downstream callers: Isolated leaf function returns 0 direct and 0 indirect callers."""
    (tmp_path / "leaf").mkdir()
    (tmp_path / "leaf" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "leaf" / "isolated.py").write_text(
        "def standalone_utility():\n"
        "    return 'nobody calls me'\n",
        encoding="utf-8",
    )

    engine = RepositoryEvolutionEngine(repo_root=str(tmp_path))
    sym = ChangedSymbol(name="standalone_utility", file_path="leaf/isolated.py", line_start=1, line_end=3)
    direct, indirect = engine.build_transitive_impact_graph([sym], max_depth=3)

    assert len(direct) == 0
    assert len(indirect) == 0


def test_transitive_blast_radius_traversal_failure_handling():
    """9. Invalid git reference raises InvalidGitReferenceError instead of returning silent 0s."""
    from testpilot.evolution.engine import InvalidGitReferenceError
    engine = RepositoryEvolutionEngine(repo_root=".")
    import pytest
    with pytest.raises(InvalidGitReferenceError):
        engine.resolve_git_diff(base_ref="nonexistent_git_branch_xyz_12345")


def test_transitive_blast_radius_api_field_consistency():
    """10. API endpoint /api/evolution/analyze returns consistent direct and indirect impact fields."""
    from fastapi.testclient import TestClient

    from testpilot.web.api import app

    client = TestClient(app)
    resp = client.post(
        "/api/evolution/analyze",
        json={"base_ref": "HEAD~1", "target_ref": "HEAD", "max_depth": 3, "repo_path": "."},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "direct_impacts" in data
    assert "indirect_impacts" in data
    assert "total_impacted_symbols" in data
    assert isinstance(data["direct_impacts"], list)
    assert isinstance(data["indirect_impacts"], list)
    assert data["total_impacted_symbols"] == len(data["direct_impacts"]) + len(data["indirect_impacts"])

    for node in data["direct_impacts"]:
        assert node["depth"] == 1
        assert node["impact_type"] in ("DIRECT", "EVENT_REGISTRATION")
        assert "evidence" in node
        assert "resolution_engine" in node["evidence"]
        assert "evidence_source" in node["evidence"]

    for node in data["indirect_impacts"]:
        assert node["depth"] >= 2
        assert node["impact_type"] == "INDIRECT"
        assert "evidence" in node
        assert len(node["evidence"]["call_chain"]) >= 2


def test_evolution_analyze_nonexistent_repo_path_fallback():
    """Verify that an invalid or non-existent repo_path falls back safely without unhandled exceptions."""
    from fastapi.testclient import TestClient
    from testpilot.web.api import app

    client = TestClient(app)
    resp = client.post(
        "/api/evolution/analyze",
        json={"base_ref": "HEAD~1", "target_ref": "HEAD", "max_depth": 2, "repo_path": "/nonexistent/path/xyz_9999"},
    )
    # Backend falls back to current working directory '.' or returns graceful status
    assert resp.status_code == 200
    data = resp.json()
    assert "changed_symbols" in data


def test_evolution_analyze_empty_diff_clean_terminal_state():
    """Verify that HEAD..HEAD produces genuine zero counts without hanging."""
    from fastapi.testclient import TestClient
    from testpilot.web.api import app

    client = TestClient(app)
    resp = client.post(
        "/api/evolution/analyze",
        json={"base_ref": "HEAD", "target_ref": "HEAD", "max_depth": 2, "repo_path": "."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("changed_files", [])) == 0
    assert len(data.get("changed_symbols", [])) == 0
    assert len(data.get("direct_impacts", [])) == 0
    assert len(data.get("indirect_impacts", [])) == 0
    assert len(data.get("prioritized_tests", [])) == 0
    assert len(data.get("semantic_impacts", [])) == 0
