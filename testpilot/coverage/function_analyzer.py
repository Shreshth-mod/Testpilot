"""
Function-Level Coverage and Uncovered Code Analysis Subsystem for TestPilot AI.
Converts empirical runtime line coverage into AST-grounded function, method,
and class coverage metrics, avoiding comment/blank line distortion.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Optional

from testpilot.coverage.models import (
    CoverageReport,
    FileCoverageSummary,
    FunctionCoverage,
    FunctionStatus,
    LineCoverageData,
    RepositoryCoverageSummary,
)

logger = logging.getLogger("testpilot.coverage.function_analyzer")


class FunctionDefVisitor(ast.NodeVisitor):
    """
    AST visitor that traverses Python source code and extracts all top-level functions,
    class methods, and nested functions with exact statement line ranges.
    """

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.functions: list[dict[str, Any]] = []
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        class_name = self._class_stack[-1] if self._class_stack else None
        is_nested = len(self._func_stack) > 0

        # Build qualified name
        if class_name:
            qual_name = f"{class_name}.{node.name}"
        elif self._func_stack:
            qual_name = f"{'.'.join(self._func_stack)}.<locals>.{node.name}"
        else:
            qual_name = node.name

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        # Collect body statement line ranges (excluding top-level docstrings)
        body_statement_ranges: list[tuple[int, int]] = []
        is_first = True
        for stmt in node.body:
            if is_first and isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str):
                is_first = False
                continue
            is_first = False
            b_start = stmt.lineno
            b_end = getattr(stmt, "end_lineno", stmt.lineno)
            body_statement_ranges.append((b_start, b_end))

        # Collect child function line ranges to exclude from parent's direct lines
        nested_ranges: list[tuple[int, int]] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c_start = child.lineno
                c_end = getattr(child, "end_lineno", child.lineno)
                nested_ranges.append((c_start, c_end))

        self.functions.append({
            "name": node.name,
            "qualified_name": qual_name,
            "class_name": class_name,
            "start_line": start_line,
            "end_line": end_line,
            "is_nested": is_nested,
            "body_statement_ranges": body_statement_ranges,
            "nested_ranges": nested_ranges,
            "node": node,
        })

        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()


class FunctionCoverageAnalyzer:
    """
    Analyzes line coverage data against Python AST structures to produce
    comprehensive function-level coverage metrics and identify uncovered code.
    """

    @classmethod
    def analyze_file(
        cls,
        file_path: Path | str,
        line_cov: LineCoverageData,
        repo_root: Optional[Path | str] = None,
    ) -> FileCoverageSummary:
        """
        Calculates function-level coverage for a single Python source file.
        """
        fpath = Path(file_path)
        if not fpath.is_absolute() and repo_root:
            abs_path = Path(repo_root) / fpath
        else:
            abs_path = fpath

        rel_path = line_cov.file_path or str(file_path).replace("\\", "/")

        if not abs_path.exists():
            return FileCoverageSummary(
                file_path=rel_path,
                total_statements=line_cov.total_statements,
                covered_statements=line_cov.covered_statements,
                missing_statements=len(line_cov.missing_lines),
                line_coverage_pct=line_cov.line_coverage_pct,
            )

        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read source file {abs_path}: {e}")
            return FileCoverageSummary(
                file_path=rel_path,
                total_statements=line_cov.total_statements,
                covered_statements=line_cov.covered_statements,
                missing_statements=len(line_cov.missing_lines),
                line_coverage_pct=line_cov.line_coverage_pct,
            )

        source_lines = content.splitlines()

        try:
            tree = ast.parse(content, filename=str(abs_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {rel_path} for function coverage: {e}")
            return FileCoverageSummary(
                file_path=rel_path,
                total_statements=line_cov.total_statements,
                covered_statements=line_cov.covered_statements,
                missing_statements=len(line_cov.missing_lines),
                line_coverage_pct=line_cov.line_coverage_pct,
            )

        visitor = FunctionDefVisitor(rel_path, source_lines)
        visitor.visit(tree)

        exec_set = set(line_cov.executable_lines)
        done_set = set(line_cov.executed_lines)
        miss_set = set(line_cov.missing_lines)

        function_coverages: list[FunctionCoverage] = []

        for fn_info in visitor.functions:
            s_line = fn_info["start_line"]
            e_line = fn_info["end_line"]

            # Filter executable lines belonging to this function
            # Exclude lines belonging to nested functions so parent isn't unfairly penalized
            nested_lines = set()
            for n_start, n_end in fn_info["nested_ranges"]:
                for nl in range(n_start, n_end + 1):
                    nested_lines.add(nl)

            # Collect body executable statements
            body_exec = []
            for b_start, b_end in fn_info["body_statement_ranges"]:
                for l in exec_set:
                    if b_start <= l <= b_end and l not in nested_lines:
                        body_exec.append(l)
            body_exec = sorted(list(set(body_exec)))

            if body_exec:
                body_executed = sorted([l for l in body_exec if l in done_set])
                body_missing = sorted([l for l in body_exec if l in miss_set])

                if len(body_executed) == 0:
                    status = FunctionStatus.UNCOVERED
                    pct = 0.0
                elif len(body_missing) == 0:
                    status = FunctionStatus.FULLY_COVERED
                    pct = 100.0
                else:
                    status = FunctionStatus.PARTIALLY_COVERED
                    pct = round((len(body_executed) / len(body_exec)) * 100.0, 1)

                fn_executable = body_exec
                fn_executed = body_executed
                fn_missing = body_missing
            else:
                # Function has no executable body statements (e.g. empty pass or protocol stub)
                if s_line in done_set:
                    status = FunctionStatus.FULLY_COVERED
                    pct = 100.0
                else:
                    status = FunctionStatus.UNCOVERED
                    pct = 0.0
                fn_executable = [s_line] if s_line in exec_set else []
                fn_executed = [s_line] if s_line in done_set else []
                fn_missing = [s_line] if s_line in miss_set else []

            function_coverages.append(
                FunctionCoverage(
                    function_name=fn_info["name"],
                    qualified_name=fn_info["qualified_name"],
                    file_path=rel_path,
                    class_name=fn_info["class_name"],
                    start_line=s_line,
                    end_line=e_line,
                    executable_lines=fn_executable,
                    executed_lines=fn_executed,
                    missing_lines=fn_missing,
                    coverage_pct=pct,
                    status=status,
                    is_nested=fn_info["is_nested"],
                )
            )

        total_funcs = len(function_coverages)
        covered_funcs = sum(1 for f in function_coverages if f.status == FunctionStatus.FULLY_COVERED)
        partial_funcs = sum(1 for f in function_coverages if f.status == FunctionStatus.PARTIALLY_COVERED)
        uncovered_funcs = sum(1 for f in function_coverages if f.status == FunctionStatus.UNCOVERED)

        func_pct = round((covered_funcs / max(1, total_funcs)) * 100.0, 1) if total_funcs > 0 else 100.0

        return FileCoverageSummary(
            file_path=rel_path,
            total_statements=line_cov.total_statements,
            covered_statements=line_cov.covered_statements,
            missing_statements=len(line_cov.missing_lines),
            line_coverage_pct=line_cov.line_coverage_pct,
            total_functions=total_funcs,
            covered_functions=covered_funcs,
            partially_covered_functions=partial_funcs,
            uncovered_functions=uncovered_funcs,
            function_coverage_pct=func_pct,
            functions=function_coverages,
        )

    @classmethod
    def analyze_report(
        cls,
        report: CoverageReport,
        repo_path: Optional[Path | str] = None,
    ) -> CoverageReport:
        """
        Enriches a CoverageReport with function-level metrics across all tracked files.
        """
        if not report.files:
            return report
        root = Path(repo_path or report.repo_path).expanduser().resolve()
        file_summaries: list[FileCoverageSummary] = []

        total_functions = 0
        covered_functions = 0
        partially_covered_functions = 0
        uncovered_functions = 0

        for file_path, line_cov in report.files.items():
            f_sum = cls.analyze_file(
                file_path=file_path,
                line_cov=line_cov,
                repo_root=root,
            )
            file_summaries.append(f_sum)

            total_functions += f_sum.total_functions
            covered_functions += f_sum.covered_functions
            partially_covered_functions += f_sum.partially_covered_functions
            uncovered_functions += f_sum.uncovered_functions

        func_cov_pct = (
            round((covered_functions / max(1, total_functions)) * 100.0, 1)
            if total_functions > 0 else 100.0
        )

        report.file_summaries = file_summaries
        report.total_functions = total_functions
        report.covered_functions = covered_functions
        report.partially_covered_functions = partially_covered_functions
        report.uncovered_functions = uncovered_functions
        report.function_coverage_pct = func_cov_pct

        return report

    @classmethod
    def get_summary(cls, report: CoverageReport) -> RepositoryCoverageSummary:
        """Generates a high-level RepositoryCoverageSummary from an analyzed CoverageReport."""
        return RepositoryCoverageSummary(
            repo_id=report.repo_id,
            commit_hash=report.commit_hash,
            workflow_id=report.workflow_id,
            total_files=len(report.files),
            total_statements=report.total_statements,
            covered_statements=report.covered_statements,
            missing_statements=report.missing_statements,
            overall_line_coverage_pct=report.line_coverage_pct,
            total_functions=report.total_functions,
            covered_functions=report.covered_functions,
            partially_covered_functions=report.partially_covered_functions,
            uncovered_functions=report.uncovered_functions,
            overall_function_coverage_pct=report.function_coverage_pct,
        )
