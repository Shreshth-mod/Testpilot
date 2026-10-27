"""
AST & Code Intelligence Engine for TestPilot AI.
Extracts function signatures, parameters, docstrings, branch conditions,
and boundary candidates directly from source files and unified git diffs.
"""

import ast
import re
from pathlib import Path
from typing import Optional

from testpilot.core.models import (
    ASTFunctionDef,
    BoundaryCandidate,
    DiffAnalysis,
    ParameterInfo,
)


class ASTDiffParser:
    """
    Parses Python source code and unified git diffs using AST analysis
    to uncover function structures, cyclomatic branch nodes, and boundary values.
    """

    @classmethod
    def parse_file(cls, file_path: str) -> list[ASTFunctionDef]:
        """Parse all function and method definitions in a Python file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        source_code = path.read_text(encoding="utf-8")
        return cls.parse_source(source_code, file_path=str(path))

    @classmethod
    def parse_source(cls, source_code: str, file_path: str = "<memory>") -> list[ASTFunctionDef]:
        """Parse AST of Python source string and extract all function definitions."""
        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            # Python < 3.12 compatibility fallback: sanitize PEP 695 type alias statements, generic parameters, and unparenthesized exceptions
            out_lines = []
            for line in source_code.splitlines():
                if re.match(r"^\s*type\s+[A-Za-z0-9_]+", line):
                    m = re.match(r"^(\s*)type\s+([A-Za-z0-9_]+)", line)
                    out_lines.append(f"{m.group(1)}{m.group(2)} = None" if m else line)
                else:
                    out_lines.append(line)
            sanitized = "\n".join(out_lines)
            sanitized = re.sub(r"(\b(?:def|class)\s+[A-Za-z0-9_]+)\[(?:[^\[\]]|\[[^\[\]]*\])*\]", r"\1", sanitized)
            sanitized = re.sub(r"(?m)^(\s*except\s+)([A-Za-z0-9_.]+(?:\s*,\s*[A-Za-z0-9_.]+)+)\s*:", r"\1(\2):", sanitized)
            try:
                tree = ast.parse(sanitized, filename=file_path)
            except SyntaxError:
                raise ValueError(f"Failed to parse AST for {file_path}: {e}") from e

        source_lines = source_code.splitlines()
        extracted: list[ASTFunctionDef] = []

        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.class_stack: list[str] = []

            @property
            def current_class(self) -> Optional[str]:
                return ".".join(self.class_stack) if self.class_stack else None

            def visit_ClassDef(self, node: ast.ClassDef):
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                func_def = cls._extract_function_def(node, source_lines, file_path, self.current_class)
                extracted.append(func_def)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                func_def = cls._extract_function_def(node, source_lines, file_path, self.current_class)
                extracted.append(func_def)
                self.generic_visit(node)

        visitor = FunctionVisitor()
        visitor.visit(tree)
        return extracted

    @classmethod
    def _extract_function_def(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        file_path: str,
        class_name: Optional[str],
    ) -> ASTFunctionDef:
        docstring = ast.get_docstring(node)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno

        raw_source = "\n".join(source_lines[line_start - 1 : line_end])

        # Extract parameters
        parameters: list[ParameterInfo] = []
        pos_args = node.args.args
        defaults = [None] * (len(pos_args) - len(node.args.defaults)) + node.args.defaults

        for arg, default in zip(pos_args, defaults, strict=False):
            if arg.arg in ("self", "cls"):
                continue

            type_annot = ast.unparse(arg.annotation) if arg.annotation else None
            default_val = ast.unparse(default) if default is not None else None
            is_req = default is None

            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    type_annotation=type_annot,
                    default_value=default_val,
                    is_required=is_req,
                )
            )

        # Return type
        return_type = ast.unparse(node.returns) if node.returns else None

        # Branch conditions and comparison constants
        branch_conditions: list[str] = []
        boundary_candidates: list[BoundaryCandidate] = []

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While)):
                try:
                    cond_str = ast.unparse(child.test)
                    branch_conditions.append(cond_str)
                except Exception:
                    pass

            if isinstance(child, ast.Compare):
                # Inspect comparisons for boundary values (e.g. subtotal >= 50.0)
                cls._extract_compare_boundaries(child, parameters, boundary_candidates)

        # Generate generic boundary candidates for parameters
        for param in parameters:
            cls._generate_parameter_boundaries(param, boundary_candidates)

        # Extract decorators
        decorators: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        return ASTFunctionDef(
            name=node.name,
            class_name=class_name,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            branch_conditions=branch_conditions,
            boundary_candidates=boundary_candidates,
            raw_source=raw_source,
            decorators=decorators,
        )

    @classmethod
    def _extract_compare_boundaries(
        cls,
        compare_node: ast.Compare,
        parameters: list[ParameterInfo],
        candidates: list[BoundaryCandidate],
    ):
        """Extract explicit boundary numbers from AST Compare nodes (e.g., x >= 50)."""
        left_name = ast.unparse(compare_node.left)
        for op, comparator in zip(compare_node.ops, compare_node.comparators, strict=False):
            comp_str = ast.unparse(comparator)
            try:
                # If comparator is a literal number
                val = ast.literal_eval(comparator)
                if isinstance(val, (int, float)):
                    # Boundary values are val - 1 (or val - 0.01), val, val + 1
                    delta = 0.01 if isinstance(val, float) else 1
                    candidates.append(
                        BoundaryCandidate(
                            parameter_name=left_name,
                            boundary_type="exact_threshold",
                            suggested_value=val,
                            rationale=f"Exact decision boundary from condition '{left_name} {op.__class__.__name__} {comp_str}'",
                        )
                    )
                    candidates.append(
                        BoundaryCandidate(
                            parameter_name=left_name,
                            boundary_type="threshold_minus_epsilon",
                            suggested_value=round(val - delta, 4),
                            rationale=f"Just below decision boundary from '{left_name} {op.__class__.__name__} {comp_str}'",
                        )
                    )
                    candidates.append(
                        BoundaryCandidate(
                            parameter_name=left_name,
                            boundary_type="threshold_plus_epsilon",
                            suggested_value=round(val + delta, 4),
                            rationale=f"Just above decision boundary from '{left_name} {op.__class__.__name__} {comp_str}'",
                        )
                    )
            except Exception:
                pass

    @classmethod
    def _generate_parameter_boundaries(
        cls,
        param: ParameterInfo,
        candidates: list[BoundaryCandidate],
    ):
        """Generate canonical boundary values based on parameter type annotation."""
        type_str = (param.type_annotation or "").lower()

        if "float" in type_str or "int" in type_str:
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="zero",
                    suggested_value=0.0 if "float" in type_str else 0,
                    rationale=f"Boundary partition: zero value for {param.name}",
                )
            )
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="negative",
                    suggested_value=-1.0 if "float" in type_str else -1,
                    rationale=f"Negative boundary test to verify sign validation for {param.name}",
                )
            )
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="large_value",
                    suggested_value=999999.0 if "float" in type_str else 999999,
                    rationale=f"Stress boundary partition: high magnitude for {param.name}",
                )
            )

        if "str" in type_str:
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="empty_string",
                    suggested_value="",
                    rationale=f"Empty string boundary value for {param.name}",
                )
            )
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="whitespace_string",
                    suggested_value="   ",
                    rationale=f"Whitespace-only boundary value for {param.name}",
                )
            )

        if "list" in type_str:
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="empty_list",
                    suggested_value=[],
                    rationale=f"Empty array boundary partition for {param.name}",
                )
            )

        if "optional" in type_str or "none" in type_str:
            candidates.append(
                BoundaryCandidate(
                    parameter_name=param.name,
                    boundary_type="null_value",
                    suggested_value=None,
                    rationale=f"Null / None boundary partition for {param.name}",
                )
            )

    @classmethod
    def parse_diff(cls, diff_text: str, repo_root: str = ".") -> DiffAnalysis:
        """
        Parses unified git diff output, identifies modified line ranges per file,
        and extracts AST function definitions that intersect with the changes.
        """
        modified_files: list[str] = []
        modified_functions: list[ASTFunctionDef] = []
        file_chunks: dict[str, list[tuple[int, int]]] = {}

        current_file: Optional[str] = None
        hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                raw_path = line[6:].strip()
                if raw_path.endswith(".py"):
                    current_file = raw_path
                    modified_files.append(current_file)
                    file_chunks[current_file] = []
                else:
                    current_file = None
            elif current_file and line.startswith("@@"):
                match = hunk_re.match(line)
                if match:
                    start_line = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    file_chunks[current_file].append((start_line, start_line + count))

        root = Path(repo_root)
        for rel_path, line_ranges in file_chunks.items():
            full_path = root / rel_path
            if not full_path.exists():
                continue
            all_funcs = cls.parse_file(str(full_path))
            for f in all_funcs:
                # Check if function lines overlap with diff chunks
                is_modified = any(
                    not (f.line_end < chunk_start or f.line_start > chunk_end)
                    for chunk_start, chunk_end in line_ranges
                )
                if is_modified:
                    modified_functions.append(f)

        total_boundaries = sum(len(f.boundary_candidates) for f in modified_functions)

        return DiffAnalysis(
            modified_files=modified_files,
            modified_functions=modified_functions,
            total_boundary_points=total_boundaries,
            diff_text=diff_text,
        )
