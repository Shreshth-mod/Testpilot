"""
Test-to-Code Mapping and Impact Graph Integration Subsystem for TestPilot AI.
Establishes empirical execution linkages between individual test cases, source files,
functions, and statements, integrating with AST impact graphs and test selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional
import uuid

import coverage

from testpilot.coverage.function_analyzer import FunctionCoverageAnalyzer
from testpilot.coverage.models import (
    CoverageEvidenceType,
    CoverageReport,
    LineCoverageData,
    TestToCodeMapping,
)
from testpilot.coverage.storage import CoverageStore

logger = logging.getLogger("testpilot.coverage.test_mapping")


class TestToCodeMapper:
    """
    Maps individual test executions to the exact source files and functions they exercise,
    providing empirical evidence distinct from static AST reachability or RAG similarity.
    """

    def __init__(self, store: Optional[CoverageStore] = None) -> None:
        self.store = store or CoverageStore.get_instance()

    def map_test_case(
        self,
        repo_path: Path | str,
        test_case: dict[str, Any],
        source_dirs: Optional[list[str]] = None,
        timeout_seconds: int = 30,
    ) -> TestToCodeMapping:
        """
        Executes a single test case under coverage.py supervision to determine
        the exact source files, functions, and lines it executes.
        """
        abs_repo = Path(repo_path).expanduser().resolve()
        t_name = test_case.get("test_name", "test")
        t_file = test_case.get("test_file", "")
        test_id = f"{t_file}::{t_name}" if "::" not in t_name else t_name

        tmp_dir = abs_repo / ".testpilot" / "tmp_coverage"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_file = tmp_dir / f"test_{uuid.uuid4().hex[:8]}.coverage"

        env = dict(sys.modules.get("os", {}).environ if hasattr(sys.modules.get("os"), "environ") else {})
        import os
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        pp_parts = [str(abs_repo)]
        src_path = abs_repo / "src"
        if src_path.is_dir():
            pp_parts.insert(0, str(src_path))
        if "PYTHONPATH" in env and env["PYTHONPATH"]:
            pp_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pp_parts)

        python_bin = sys.executable or "python"
        cmd = [
            python_bin, "-m", "coverage", "run",
            f"--data-file={str(data_file)}",
            f"--source={str(abs_repo)}",
            "--omit=*/tests/*,*test_*.py,*_test.py,*/.venv/*,*/venv/*,conftest.py,setup.py",
            "-m", "pytest",
            t_file,
            "-k", t_name,
            "-v",
        ]

        covered_files: list[str] = []
        covered_functions: list[str] = []
        lines_by_file: dict[str, list[int]] = {}

        try:
            res = subprocess.run(
                cmd,
                cwd=str(abs_repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )

            if data_file.exists():
                cov = coverage.Coverage(data_file=str(data_file))
                cov.load()
                cov_data = cov.get_data()

                for abs_file in cov_data.measured_files():
                    p_file = Path(abs_file)
                    try:
                        rel_file = str(p_file.relative_to(abs_repo)).replace("\\", "/")
                    except ValueError:
                        continue

                    # Omit tests
                    rel_lower = rel_file.lower()
                    if rel_lower.startswith("tests/") or "/tests/" in rel_lower or rel_lower.startswith("test_") or "/test_" in rel_lower:
                        continue

                    analysis = cov.analysis2(abs_file)
                    executable = set(analysis[1])
                    missing = set(analysis[3])
                    executed = sorted(list(executable - missing))

                    if executed:
                        covered_files.append(rel_file)
                        lines_by_file[rel_file] = executed

                        # Determine which functions were touched
                        f_cov = LineCoverageData(
                            file_path=rel_file,
                            executable_lines=sorted(list(executable)),
                            executed_lines=executed,
                            missing_lines=sorted(list(missing)),
                            total_statements=len(executable),
                            covered_statements=len(executed),
                        )
                        f_sum = FunctionCoverageAnalyzer.analyze_file(
                            file_path=rel_file,
                            line_cov=f_cov,
                            repo_root=abs_repo,
                        )
                        for fn in f_sum.functions:
                            # If function has executed lines, this test ran it!
                            if fn.executed_lines:
                                covered_functions.append(fn.qualified_name)
        except Exception as e:
            logger.warning(f"Failed to map test {test_id}: {e}")
        finally:
            if data_file.exists():
                try:
                    data_file.unlink()
                except Exception:
                    pass

        return TestToCodeMapping(
            test_id=test_id,
            test_name=t_name,
            test_file=t_file,
            covered_files=sorted(list(set(covered_files))),
            covered_functions=sorted(list(set(covered_functions))),
            executed_lines_by_file=lines_by_file,
            evidence_type=CoverageEvidenceType.RUNTIME_COVERAGE,
        )

    def map_all_tests(
        self,
        repo_path: Path | str,
        test_cases: list[dict[str, Any]],
        timeout_per_test: int = 30,
    ) -> list[TestToCodeMapping]:
        """Maps each test case in the list to its touched source files and functions."""
        mappings: list[TestToCodeMapping] = []
        for tc in test_cases:
            m = self.map_test_case(repo_path, tc, timeout_seconds=timeout_per_test)
            mappings.append(m)
        return mappings

    def attach_mappings_to_report(
        self,
        report: CoverageReport,
        mappings: list[TestToCodeMapping],
    ) -> CoverageReport:
        """
        Attaches test-to-code mappings to a CoverageReport and updates
        each FunctionCoverage entry with its executing test IDs.
        """
        report.test_mappings = mappings

        # Build index: qualified_name -> list of test_ids
        func_to_tests: dict[str, set[str]] = {}
        for m in mappings:
            for fn_name in m.covered_functions:
                func_to_tests.setdefault(fn_name, set()).add(m.test_id)

        # Update executing_tests in file_summaries
        for fsum in report.file_summaries:
            for fn in fsum.functions:
                if fn.qualified_name in func_to_tests:
                    fn.executing_tests = sorted(list(func_to_tests[fn.qualified_name]))

        self.store.save_report(report)
        return report

    def build_integrated_impact_graph(
        self,
        report: CoverageReport,
        static_ast_edges: Optional[list[dict[str, Any]]] = None,
        rag_similarity_edges: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Constructs an integrated evidence graph displaying:
        1. RUNTIME_COVERAGE (Empirically verified execution with confidence 1.0)
        2. DIRECT_AST_CALL (Static AST call site with confidence 0.85)
        3. TRANSITIVE_CHAIN (Multi-hop static call chain with confidence 0.60)
        4. SEMANTIC_SIMILARITY (RAG vector heuristic with confidence 0.40)
        """
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        # 1. Add Function Nodes
        for fsum in report.file_summaries:
            for fn in fsum.functions:
                node_id = f"func:{fn.qualified_name}"
                nodes[node_id] = {
                    "node_id": node_id,
                    "label": fn.qualified_name,
                    "type": "FUNCTION",
                    "file_path": fn.file_path,
                    "coverage_pct": fn.coverage_pct,
                    "status": fn.status.value,
                    "executing_tests_count": len(fn.executing_tests),
                }

        # 2. Add Test Nodes and Runtime Coverage Edges
        for tm in report.test_mappings:
            test_node_id = f"test:{tm.test_id}"
            nodes[test_node_id] = {
                "node_id": test_node_id,
                "label": tm.test_name,
                "type": "TEST",
                "test_file": tm.test_file,
                "covered_functions_count": len(tm.covered_functions),
            }

            for fn_name in tm.covered_functions:
                target_node_id = f"func:{fn_name}"
                if target_node_id not in nodes:
                    nodes[target_node_id] = {
                        "node_id": target_node_id,
                        "label": fn_name,
                        "type": "FUNCTION",
                        "coverage_pct": 100.0,
                        "status": "FULLY_COVERED",
                        "executing_tests_count": 1,
                    }
                edges.append({
                    "source": test_node_id,
                    "target": target_node_id,
                    "evidence_type": CoverageEvidenceType.RUNTIME_COVERAGE.value,
                    "confidence": 1.0,
                    "is_empirical": True,
                    "description": f"Test '{tm.test_name}' empirically executed '{fn_name}' in runtime container/subprocess.",
                })

        # 3. Add Static AST Edges if provided
        for se in (static_ast_edges or []):
            s_node = se.get("source")
            t_node = se.get("target")
            edges.append({
                "source": s_node,
                "target": t_node,
                "evidence_type": se.get("evidence_type", "DIRECT_AST_CALL"),
                "confidence": se.get("confidence", 0.85),
                "is_empirical": False,
                "description": se.get("description", "Static AST dependency"),
            })

        # 4. Add RAG Similarity Edges if provided
        for re in (rag_similarity_edges or []):
            s_node = re.get("source")
            t_node = re.get("target")
            edges.append({
                "source": s_node,
                "target": t_node,
                "evidence_type": CoverageEvidenceType.SEMANTIC_SIMILARITY.value,
                "confidence": re.get("confidence", 0.40),
                "is_empirical": False,
                "description": re.get("description", "Semantic embedding vector similarity"),
            })

        runtime_edges_count = sum(1 for e in edges if e.get("evidence_type") == "RUNTIME_COVERAGE")
        ast_edges_count = sum(1 for e in edges if "AST" in e.get("evidence_type", ""))
        rag_edges_count = sum(1 for e in edges if e.get("evidence_type") == "SEMANTIC_SIMILARITY")

        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": list(nodes.values()),
            "edges": edges,
            "evidence_breakdown": {
                "runtime_confirmed": runtime_edges_count,
                "ast_inferred": ast_edges_count,
                "semantic_similarity": rag_edges_count,
            },
        }

