"""
Unit tests for Sourcegraph client and Local AST Fallback.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from testpilot.sourcegraph.client import LocalCodeGraphFallback, SourcegraphClient
from testpilot.web.api import app


def test_sourcegraph_client_fallback_mode():
    client = SourcegraphClient(endpoint="http://localhost:9999/.api/graphql", timeout=0.1)
    # When Sourcegraph is offline, is_available and is_alive return False gracefully
    assert client.is_available() is False
    assert client.is_alive() is False


def test_local_fallback_finds_callers_of_order_service():
    fallback = LocalCodeGraphFallback(repo_root=".")
    callers = fallback.find_callers("calculate_order_totals")

    # In testbed/app/main.py, create_order calls OrderService.calculate_order_totals
    assert len(callers) >= 1
    caller_names = [c["caller_name"] for c in callers]
    file_paths = [c["file_path"] for c in callers]

    assert "create_order" in caller_names
    assert any("testbed/app/main.py" in p for p in file_paths)


def test_local_fallback_finds_status_transition_callers():
    fallback = LocalCodeGraphFallback(repo_root=".")
    callers = fallback.find_callers("transition_order_status")

    assert len(callers) >= 1
    caller_names = [c["caller_name"] for c in callers]
    assert "transition_order_status" in caller_names


def test_sourcegraph_definitions_and_references():
    client = SourcegraphClient(repo_root=".")
    defs = client.find_definitions("calculate_order_totals")
    assert isinstance(defs, list)
    assert len(defs) >= 1
    assert any("order_service.py" in d["file_path"] for d in defs)

    refs = client.find_references("calculate_order_totals")
    assert isinstance(refs, list)
    assert len(refs) >= 1


def test_sourcegraph_test_references():
    client = SourcegraphClient(repo_root=".")
    test_refs = client.find_test_references("calculate_order_totals")
    assert isinstance(test_refs, list)
    # References should be from test files
    for r in test_refs:
        assert "test" in r["file_path"].lower()


def test_sourcegraph_class_usages():
    client = SourcegraphClient(repo_root=".")
    usages = client.find_class_usages("OrderService")
    assert isinstance(usages, list)
    assert len(usages) >= 1


def test_sourcegraph_normalize_evidence():
    client = SourcegraphClient(repo_root=".")
    # In offline fallback mode:
    ev = client.normalize_evidence(
        query="type:references calculate_order_totals",
        symbol="calculate_order_totals",
        results=[{"caller_name": "create_order", "file_path": "testbed/app/main.py", "source_type": "local_ast_fallback"}],
    )
    assert ev["symbol"] == "calculate_order_totals"
    assert ev["has_evidence"] is True
    assert ev["availability"] == "UNAVAILABLE"
    assert ev["status"] == "SOURCEGRAPH_UNAVAILABLE_FALLBACK"
    assert ev["results_count"] == 1


def test_sourcegraph_mocked_online_flow(monkeypatch):
    client = SourcegraphClient()
    monkeypatch.setattr(client, "is_available", lambda: True)

    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "data": {
                    "search": {
                        "results": {
                            "results": [
                                {
                                    "file": {"path": "testbed/app/routers/orders.py"},
                                    "symbols": [
                                        {
                                            "name": "calculate_order_totals",
                                            "kind": "FUNCTION",
                                            "location": {"range": {"start": {"line": 42}}},
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                }
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())

    defs = client.find_definitions("calculate_order_totals")
    assert len(defs) == 1
    assert defs[0]["source_type"] == "sourcegraph_graphql"
    assert defs[0]["line_number"] == 42


# =============================================================================
# Code Intelligence UI & API Test Suite (Requirements 1-12)
# =============================================================================


def test_1_symbol_search():
    client = SourcegraphClient(repo_root=".")
    res = client.search("calculate_order_totals", search_type="symbol")
    assert res["query"] == "calculate_order_totals"
    assert "definitions" in res
    assert "references" in res
    assert "relationships" in res
    assert res["counts"]["total"] >= 1
    assert len(res["definitions"]) >= 1
    assert res["definitions"][0]["symbol_name"] == "calculate_order_totals"


def test_2_function_search():
    client = SourcegraphClient(repo_root=".")
    funcs = client.search_functions("calculate_order_totals")
    assert isinstance(funcs, list)
    assert len(funcs) >= 1
    assert funcs[0]["kind"] == "FUNCTION"
    assert "order_service.py" in funcs[0]["file_path"]


def test_3_definition_search():
    client = SourcegraphClient(repo_root=".")
    defs = client.find_definitions("calculate_order_totals")
    assert len(defs) >= 1
    assert defs[0]["line_number"] > 0
    assert "qualified_symbol" in defs[0]
    assert "order_service.calculate_order_totals" in defs[0]["qualified_symbol"]


def test_4_reference_search():
    client = SourcegraphClient(repo_root=".")
    refs = client.find_references("calculate_order_totals")
    assert len(refs) >= 1
    assert all("file_path" in r and "line_number" in r for r in refs)


def test_5_test_reference_search():
    client = SourcegraphClient(repo_root=".")
    t_refs = client.find_test_references("calculate_order_totals")
    assert len(t_refs) >= 1
    for r in t_refs:
        assert "test" in r["file_path"].lower()


def test_6_class_usage_search():
    client = SourcegraphClient(repo_root=".")
    usages = client.find_class_usages("OrderService")
    assert len(usages) >= 1
    assert any("order_service" in u["file_path"].lower() or "test" in u["file_path"].lower() for u in usages)


def test_7_sourcegraph_unavailable_mode(monkeypatch):
    client = SourcegraphClient()
    monkeypatch.setattr(client, "is_available", lambda: False)
    monkeypatch.setattr(client, "is_alive", lambda: False)
    assert client.is_available() is False
    assert client.is_alive() is False

    res = client.search("calculate_order_totals")
    assert res["engine_status"] == "FALLBACK"
    assert res["source"] == "local_ast_fallback"
    assert "Local AST Fallback" in res["engine_label"]


def test_8_local_ast_fallback_behavior():
    fallback = LocalCodeGraphFallback(repo_root=".")
    defs = fallback.find_definitions("calculate_order_totals")
    assert len(defs) >= 1
    assert defs[0]["source_type"] == "local_ast_fallback"
    assert "AST" in defs[0]["evidence_reason"]
    assert defs[0]["qualified_symbol"].endswith("calculate_order_totals")


def test_9_empty_search_result():
    client = SourcegraphClient(repo_root=".")
    nonexistent = "nonexistent_sym_" + "does_not_exist_98765"
    res = client.search(nonexistent)
    assert res["counts"]["total"] == 0
    assert len(res["definitions"]) == 0
    assert len(res["references"]) == 0
    assert len(res["test_references"]) == 0
    assert len(res["class_usages"]) == 0
    assert len(res["code_matches"]) == 0


def test_10_invalid_or_empty_search():
    client = SourcegraphClient(repo_root=".")
    res_empty = client.search("   ")
    assert res_empty["counts"]["total"] == 0
    assert res_empty["definitions"] == []

    # Safe snippet retrieval on nonexistent file
    snip = client.read_source_snippet("nonexistent/path/does_not_exist.py", 10)
    assert "File not found" in snip["code"]
    assert snip["total_lines"] == 0


def test_11_api_response_structure():
    api_client = TestClient(app)

    # Removed endpoints should return 404
    r_status = api_client.get("/api/code-intel/status")
    assert r_status.status_code == 404

    r_search = api_client.get("/api/code-intel/search?query=calculate_order_totals&search_type=symbol")
    assert r_search.status_code == 404

    r_src = api_client.get("/api/code-intel/source?file_path=testbed/app/services/order_service.py&line_number=77")
    assert r_src.status_code == 404

    # Preserved Impact Analysis callers endpoints return 200
    r_impact_callers = api_client.get("/api/impact/callers?symbol=calculate_order_totals")
    assert r_impact_callers.status_code == 200
    assert r_impact_callers.json()["symbol"] == "calculate_order_totals"
    assert "callers" in r_impact_callers.json()

    r_compat_callers = api_client.get("/api/code-intel/callers?symbol=calculate_order_totals")
    assert r_compat_callers.status_code == 200
    assert r_compat_callers.json()["symbol"] == "calculate_order_totals"

    # Search via sourcegraph search endpoint returns 200
    r_sg_search = api_client.get("/api/sourcegraph/search?query=calculate_order_totals&search_type=symbol")
    assert r_sg_search.status_code == 200
    s_json = r_sg_search.json()
    assert s_json["query"] == "calculate_order_totals"
    assert "definitions" in s_json
    assert "references" in s_json
    assert "relationships" in s_json
    assert "counts" in s_json
    assert "total_matches" in s_json

    # Dedicated category endpoints
    r_defs = api_client.get("/api/sourcegraph/definitions?symbol=calculate_order_totals")
    assert r_defs.status_code == 200
    assert r_defs.json()["total"] >= 1

    r_refs = api_client.get("/api/sourcegraph/references?symbol=calculate_order_totals")
    assert r_refs.status_code == 200
    assert r_refs.json()["total"] >= 1


def test_12_ui_rendering_structure():
    html_file = Path("testpilot/web/static/index.html")
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")

    # Code Intelligence elements must be completely removed
    assert 'id="tab-btn-code-intel"' not in content
    assert 'id="tab-code-intel"' not in content
    assert 'id="code-intel-status-pill"' not in content
    assert 'id="code-intel-query"' not in content
    assert 'id="code-intel-type"' not in content
    assert 'id="code-intel-repo"' not in content
    assert 'id="btn-code-intel-search"' not in content
    assert 'id="code-intel-results-wrapper"' not in content
    assert 'id="code-intel-source-modal"' not in content

    # All other tabs and critical UI components must remain intact
    assert 'id="tab-btn-overview"' in content
    assert 'id="tab-btn-evaluation"' in content
    assert 'id="tab-btn-evolution"' in content
    assert 'id="tab-btn-ast"' in content
    assert 'id="tab-btn-blast"' in content
    assert 'id="tab-btn-arbiter"' in content
    assert 'id="tab-overview"' in content
    assert 'id="tab-evaluation"' in content
    assert 'id="tab-evolution"' in content
    assert 'id="tab-ast"' in content
    assert 'id="tab-blast"' in content
    assert 'id="tab-arbiter"' in content
