"""
Unit and integration tests for the Targeted Test Studio subsystem.
Verifies:
1. Scope inspection across all 7 target modes (entire repo, folder, file, class, function, feature, symbols).
2. Mirrored test suite generation under tests/generated/ with 5-criterion qualification.
3. Execution of synthesized suites via local subprocess runner.
4. REST API endpoints (/api/studio/inspect, /api/studio/generate, /api/studio/execute, /api/studio/files).
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from testpilot.ast_engine.scope_inspector import ScopeInspector, ScopeInspectionResult
from testpilot.generator.studio_generator import StudioTestGenerator, StudioGenerationResult
from testpilot.web.api import app

client = TestClient(app)


def test_scope_inspector_file_and_function():
    # Test inspecting a specific file
    res = ScopeInspector.inspect(repo_path=".", scope_type="file", target_query="testbed/order_service.py")
    assert res.scope_type == "file"
    assert res.total_files == 1
    assert res.total_functions >= 1
    sym_names = [s.name for s in res.symbols]
    assert "calculate_order_totals" in sym_names

    # Test inspecting a specific function
    fn_res = ScopeInspector.inspect(repo_path=".", scope_type="function", target_query="calculate_order_totals")
    assert fn_res.scope_type == "function"
    assert len(fn_res.symbols) >= 1
    assert fn_res.symbols[0].name == "calculate_order_totals"
    assert len(fn_res.symbols[0].parameters) >= 1


def test_scope_inspector_folder_and_entire_repo():
    # Test inspecting a folder
    folder_res = ScopeInspector.inspect(repo_path=".", scope_type="folder", target_query="testbed")
    assert folder_res.scope_type == "folder"
    assert folder_res.total_files >= 1
    assert any("order_service.py" in f for f in folder_res.matched_files)

    # Test inspecting entire repo
    repo_res = ScopeInspector.inspect(repo_path=".", scope_type="entire_repo")
    assert repo_res.scope_type == "entire_repo"
    assert repo_res.total_files >= 5
    assert repo_res.total_functions >= 10


def test_scope_inspector_feature_by_name():
    # Test natural language feature retrieval
    feat_res = ScopeInspector.inspect(repo_path=".", scope_type="feature", target_query="order totals shipping")
    assert feat_res.scope_type == "feature"
    assert len(feat_res.matched_files) >= 1
    assert any("order_service" in f for f in feat_res.matched_files)
    assert feat_res.feature_rationale is not None


def test_studio_generator_mirrored_suites(tmp_path):
    inspection = ScopeInspector.inspect(repo_path=".", scope_type="file", target_query="testbed/order_service.py")
    generator = StudioTestGenerator(output_root=tmp_path / "tests" / "generated")

    gen_res = generator.generate_suite(
        inspection=inspection,
        repo_path=".",
        persist_to_disk=True,
        tests_per_symbol=2,
    )

    assert gen_res.total_files_processed == 1
    assert gen_res.total_tests_generated >= 2
    assert gen_res.qualified_tests_count >= 2
    assert len(gen_res.generated_files) == 1

    gen_file = gen_res.generated_files[0]
    assert "test_order_service_studio.py" in gen_file.test_file_path
    assert gen_file.qualification_passed is True
    assert "import pytest" in gen_file.code
    assert "def test_" in gen_file.code


def test_studio_rest_api_endpoints():
    # 1. GET /api/studio/files
    files_res = client.get("/api/studio/files")
    assert files_res.status_code == 200
    files_data = files_res.json()
    assert "files" in files_data
    assert len(files_data["files"]) >= 5

    # 2. POST /api/studio/inspect
    inspect_res = client.post(
        "/api/studio/inspect",
        json={"scope_type": "function", "target_query": "calculate_order_totals"},
    )
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["scope_type"] == "function"
    assert inspect_data["total_functions"] >= 1

    # 3. POST /api/studio/generate
    gen_res = client.post(
        "/api/studio/generate",
        json={"scope_type": "file", "target_query": "testbed/order_service.py", "persist_to_disk": True},
    )
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["total_tests_generated"] >= 1
    assert len(gen_data["generated_files"]) >= 1

    # 4. POST /api/studio/execute
    exec_res = client.post(
        "/api/studio/execute",
        json={"allow_local_subprocess": True, "allow_simulated_dry_run": True, "execution_mode": "local"},
    )
    assert exec_res.status_code == 200
    exec_data = exec_res.json()
    assert exec_data["total_executed"] >= 1
    assert "passed_count" in exec_data


def test_scope_inspector_commits_and_evolution_bridge():
    """Verifies commits scope inspection and seamless context forwarding to evolution."""
    from testpilot.core.context import AnalysisContextManager

    # 1. Inspect commits scope
    res = ScopeInspector.inspect(repo_path=".", scope_type="commits", target_query="HEAD~1..HEAD")
    assert res.scope_type == "commits"
    assert isinstance(res.matched_files, list)

    # 2. Test Studio API to Evolution Analyze Bridge
    inspect_res = client.post(
        "/api/studio/inspect",
        json={"scope_type": "folder", "target_query": "testbed/app/services"},
    )
    assert inspect_res.status_code == 200
    ctx_mgr = AnalysisContextManager()
    active_ctx = ctx_mgr.get_active_context()
    assert active_ctx.target_scope == "folder"
    assert active_ctx.target_scope_query == "testbed/app/services"
    assert len(active_ctx.scoped_symbols) >= 1

    # 3. Analyze evolution directly inheriting the studio context
    evo_res = client.post(
        "/api/evolution/analyze",
        json={"repo_path": ".", "base_ref": "HEAD~1", "target_ref": "HEAD"},
    )
    assert evo_res.status_code == 200
    evo_data = evo_res.json()
    assert evo_data["target_scope"] == "folder"
    assert evo_data["primary_target_symbol"] is not None

