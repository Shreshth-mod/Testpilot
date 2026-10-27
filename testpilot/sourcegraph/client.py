"""
Sourcegraph OSS GraphQL Client & Resilient AST Fallback for TestPilot AI.
Interfaces with Sourcegraph server (http://localhost:7080/.api/graphql) to perform
symbol cross-references and discover downstream callers (blast radius).
Includes an autonomous local AST call-graph fallback when Docker is offline.
"""

import ast
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests


def compute_qualified_symbol(file_path: str, symbol_name: str, class_name: Optional[str] = None) -> str:
    """
    Derive dot-separated qualified symbol path from file path and symbol identifier.
    e.g. django/contrib/auth/management/__init__.py + rename_permissions_after_model_rename
    -> django.contrib.auth.management.rename_permissions_after_model_rename
    """
    norm = file_path.replace("\\", "/").lstrip("./")
    if norm.endswith(".py"):
        norm = norm[:-3]
    if norm.endswith("/__init__"):
        norm = norm[:-9]
    parts = [p for p in norm.split("/") if p and p != "."]
    mod = ".".join(parts)
    if class_name and class_name != symbol_name:
        return f"{mod}.{class_name}.{symbol_name}" if mod else f"{class_name}.{symbol_name}"
    return f"{mod}.{symbol_name}" if mod else symbol_name


class LocalCodeGraphFallback:
    """
    Zero-dependency AST-driven call graph generator that indexes symbol references
    across the repository when Sourcegraph OSS is not running.
    Resolves imported symbols, aliases, and receiver modules to prevent false matches.
    """

    GENERIC_METHOD_NAMES = {
        "__init__", "setUp", "tearDown", "setUpClass", "tearDownClass", "setUpTestData",
        "setup_method", "teardown_method", "setup_class", "teardown_class",
        "run", "close", "status", "execute", "start", "stop", "reset", "clear",
        "get", "set", "update", "delete", "handle", "process", "validate",
        "save", "create", "filter", "all", "exists", "count", "send", "connect",
    }

    def __init__(self, repo_root: str = ".", secondary_roots: Optional[list[str]] = None):
        self.repo_root = Path(repo_root).resolve()
        self.secondary_roots = [Path(r).resolve() for r in secondary_roots] if secondary_roots else []
        self._file_ast_cache: dict[str, tuple[str, Optional[ast.AST]]] = {}
        self._file_imports_cache: dict[str, dict[str, dict[str, str]]] = {}
        self._query_callers_cache: dict[tuple[str, Optional[str], Optional[str]], list[dict[str, Any]]] = {}

    @staticmethod
    def _module_matches_path(module_str: str, file_path: Optional[str]) -> bool:
        """Determines if an imported module name could correspond to the given target file path."""
        if not file_path or not module_str:
            return True
        norm_path = file_path.replace("\\", "/")
        if norm_path.endswith(".py"):
            norm_path = norm_path[:-3]
        if norm_path.endswith("/__init__"):
            norm_path = norm_path[:-9]
        path_parts = [p for p in norm_path.split("/") if p and p != "."]
        mod_parts = [m for m in module_str.split(".") if m]
        if not mod_parts or not path_parts:
            return True
        mod_joined = ".".join(mod_parts)
        path_joined = ".".join(path_parts)
        if mod_joined == path_joined:
            return True
        if path_joined.startswith(mod_joined + "."):
            return True
        if mod_joined.startswith(path_joined + "."):
            return True
        if path_joined.endswith("." + mod_joined) or mod_joined.endswith("." + path_joined):
            return True
        return False

    def _get_python_files(self, root: Path) -> list[Path]:
        """Pruned, cached discovery of Python files skipping hidden dirs and virtualenvs."""
        if not hasattr(self, "_py_files_cache"):
            self._py_files_cache: dict[str, list[Path]] = {}
        r_key = str(root.resolve())
        if r_key in self._py_files_cache:
            return self._py_files_cache[r_key]

        py_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in ("venv", "env", "site-packages", "htmlcov", ".venv", "node_modules", "__pycache__")
            ]
            for f in filenames:
                if f.endswith(".py"):
                    py_files.append(Path(dirpath) / f)
        self._py_files_cache[r_key] = py_files
        return py_files

    def _find_callers_in_root(
        self,
        root: Path,
        target_function_name: str,
        target_file_path: Optional[str] = None,
        target_class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        callers: list[dict[str, Any]] = []
        norm_target_path = None
        if target_file_path:
            norm_target_path = target_file_path.replace("\\", "/").lstrip("./")

        cache_key = (target_function_name, norm_target_path, target_class_name)
        if hasattr(self, "_query_callers_cache") and cache_key in self._query_callers_cache:
            return list(self._query_callers_cache[cache_key])

        search_terms = {target_function_name}
        if target_class_name:
            search_terms.add(target_class_name)

        if not hasattr(self, "_file_ast_cache"):
            self._file_ast_cache: dict[str, tuple[str, Optional[ast.AST]]] = {}
        if not hasattr(self, "_file_imports_cache"):
            self._file_imports_cache: dict[str, dict[str, dict[str, str]]] = {}

        for py_file in self._get_python_files(root):
            file_key = str(py_file.resolve())
            cached = self._file_ast_cache.get(file_key)
            if cached is not None:
                source, tree = cached
            else:
                try:
                    source = py_file.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(source, filename=str(py_file))
                except Exception:
                    source = ""
                    tree = None
                self._file_ast_cache[file_key] = (source, tree)

            if not tree or not any(term in source for term in search_terms):
                continue

            try:
                rel_path = str(py_file.relative_to(root)).replace("\\", "/")
            except ValueError:
                rel_path = str(py_file).replace("\\", "/")

            if file_key in self._file_imports_cache:
                imported_symbols = self._file_imports_cache[file_key]
            else:
                imported_symbols = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            as_name = alias.asname or alias.name
                            imported_symbols[as_name] = {
                                "source_module": alias.name,
                                "original_symbol": alias.name.split(".")[-1],
                            }
                    elif isinstance(node, ast.ImportFrom):
                        mod_name = node.module or ""
                        for alias in node.names:
                            as_name = alias.asname or alias.name
                            imported_symbols[as_name] = {
                                "source_module": mod_name,
                                "original_symbol": alias.name,
                            }
                self._file_imports_cache[file_key] = imported_symbols

            is_same_file = bool(norm_target_path and (rel_path == norm_target_path or rel_path.endswith(norm_target_path)))

            class CallVisitor(ast.NodeVisitor):
                def __init__(self, outer_self, imported_syms, is_same, target_p, rel_p, tgt_cls):
                    self.outer = outer_self
                    self.imported_symbols = imported_syms
                    self.is_same_file = is_same
                    self.norm_target_path = target_p
                    self.rel_path = rel_p
                    self.target_class_name = tgt_cls
                    self.scope_stack: list[str] = []
                    self.class_stack: list[str] = []
                    self.local_var_types: dict[str, str] = {}

                def visit_ClassDef(self, node: ast.ClassDef):
                    self.class_stack.append(node.name)
                    self.generic_visit(node)
                    self.class_stack.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self.scope_stack.append(node.name)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self.scope_stack.append(node.name)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node: ast.Assign):
                    if isinstance(node.value, ast.Call):
                        cls_name = None
                        if isinstance(node.value.func, ast.Name):
                            cls_name = node.value.func.id
                        elif isinstance(node.value.func, ast.Attribute):
                            cls_name = node.value.func.attr
                        if cls_name:
                            for t in node.targets:
                                if isinstance(t, ast.Name):
                                    self.local_var_types[t.id] = cls_name
                    self.generic_visit(node)

                def visit_AnnAssign(self, node: ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        cls_name = None
                        if isinstance(node.annotation, ast.Name):
                            cls_name = node.annotation.id
                        elif isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                cls_name = node.value.func.id
                            elif isinstance(node.value.func, ast.Attribute):
                                cls_name = node.value.func.attr
                        if cls_name:
                            self.local_var_types[node.target.id] = cls_name
                    self.generic_visit(node)

                def visit_Call(self, node: ast.Call):
                    caller_name = self.scope_stack[-1] if self.scope_stack else "<module_level>"
                    matched = False
                    is_ambiguous = False
                    match_quality = "EXACT_AST_CALL"
                    is_constructor_target = (target_function_name == "__init__" and bool(self.target_class_name))

                    if is_constructor_target:
                        if isinstance(node.func, ast.Name):
                            called_id = node.func.id
                            if called_id == self.target_class_name:
                                if called_id in self.imported_symbols:
                                    info = self.imported_symbols[called_id]
                                    if not self.norm_target_path or self.outer._module_matches_path(info["source_module"], self.norm_target_path):
                                        matched = True
                                        match_quality = "CONFIRMED_CONSTRUCTOR_CALL"
                                elif self.is_same_file:
                                    matched = True
                                    match_quality = "SAME_FILE_CONSTRUCTOR_CALL"
                                elif not self.norm_target_path:
                                    matched = True
                                    match_quality = "CONFIRMED_CONSTRUCTOR_CALL"
                        elif isinstance(node.func, ast.Attribute):
                            attr_name = node.func.attr
                            if attr_name == self.target_class_name:
                                full_attr = self._unparse_node(node.func.value)
                                if not self.norm_target_path or self.outer._module_matches_path(full_attr, self.norm_target_path):
                                    matched = True
                                    match_quality = "CONFIRMED_CONSTRUCTOR_CALL"
                            elif attr_name == "__init__":
                                receiver = node.func.value
                                if isinstance(receiver, ast.Name) and receiver.id == self.target_class_name:
                                    matched = True
                                    match_quality = "CONFIRMED_CLASS_METHOD_CALL"
                                elif isinstance(receiver, ast.Name) and receiver.id in ("self", "cls") and self.is_same_file:
                                    if self.class_stack and self.class_stack[-1] == self.target_class_name:
                                        matched = True
                                        match_quality = "SAME_FILE_CLASS_CALL"
                    else:
                        if isinstance(node.func, ast.Name):
                            called_id = node.func.id
                            if called_id in self.imported_symbols:
                                info = self.imported_symbols[called_id]
                                if info["original_symbol"] == target_function_name:
                                    if self.norm_target_path:
                                        if self.outer._module_matches_path(info["source_module"], self.norm_target_path):
                                            matched = True
                                            match_quality = "CONFIRMED_IMPORT_CALL"
                                    else:
                                        matched = True
                                        match_quality = "CONFIRMED_IMPORT_CALL"
                            elif called_id == target_function_name:
                                if self.is_same_file:
                                    matched = True
                                    match_quality = "SAME_FILE_AST_CALL"
                                elif target_function_name in LocalCodeGraphFallback.GENERIC_METHOD_NAMES:
                                    matched = False
                                elif self.norm_target_path:
                                    matched = True
                                    is_ambiguous = True
                                    match_quality = "AMBIGUOUS_NAME_MATCH"
                                else:
                                    matched = True
                                    match_quality = "EXACT_AST_CALL"

                        elif isinstance(node.func, ast.Attribute):
                            attr_name = node.func.attr
                            if attr_name == target_function_name:
                                receiver = node.func.value
                                if isinstance(receiver, ast.Name):
                                    rec_name = receiver.id
                                    inferred_class = self.local_var_types.get(rec_name)
                                    if self.target_class_name and (rec_name == self.target_class_name or inferred_class == self.target_class_name):
                                        matched = True
                                        match_quality = "CONFIRMED_CLASS_METHOD_CALL"
                                    elif self.is_same_file and rec_name in ("self", "cls"):
                                        if not self.target_class_name or (self.class_stack and self.class_stack[-1] == self.target_class_name):
                                            matched = True
                                            match_quality = "SAME_FILE_CLASS_CALL"
                                        else:
                                            matched = False
                                    elif rec_name in self.imported_symbols:
                                        info = self.imported_symbols[rec_name]
                                        if self.norm_target_path:
                                            if self.outer._module_matches_path(info["source_module"], self.norm_target_path):
                                                matched = True
                                                match_quality = "CONFIRMED_CLASS_METHOD_CALL"
                                            else:
                                                matched = False
                                        else:
                                            matched = True
                                            match_quality = "CONFIRMED_CLASS_METHOD_CALL"
                                    else:
                                        if target_function_name in LocalCodeGraphFallback.GENERIC_METHOD_NAMES or self.target_class_name:
                                            matched = False
                                        elif self.norm_target_path:
                                            matched = True
                                            is_ambiguous = True
                                            match_quality = "AMBIGUOUS_ATTRIBUTE_CALL"
                                        else:
                                            matched = True
                                            is_ambiguous = True
                                            match_quality = "AMBIGUOUS_ATTRIBUTE_CALL"
                                elif isinstance(receiver, ast.Attribute):
                                    full_attr = self._unparse_node(receiver)
                                    if self.norm_target_path and self.outer._module_matches_path(full_attr, self.norm_target_path):
                                        matched = True
                                        match_quality = "CONFIRMED_QUALIFIED_CALL"
                                    else:
                                        if target_function_name in LocalCodeGraphFallback.GENERIC_METHOD_NAMES or self.target_class_name:
                                            matched = False
                                        else:
                                            matched = True
                                            is_ambiguous = True
                                            match_quality = "AMBIGUOUS_ATTRIBUTE_CALL"
                                else:
                                    if target_function_name in LocalCodeGraphFallback.GENERIC_METHOD_NAMES or self.target_class_name:
                                        matched = False
                                    else:
                                        matched = True
                                        is_ambiguous = True
                                        match_quality = "AMBIGUOUS_ATTRIBUTE_CALL"

                    if matched:
                        callers.append(
                            {
                                "caller_name": caller_name,
                                "class_name": self.class_stack[-1] if self.class_stack else None,
                                "file_path": self.rel_path,
                                "line_number": node.lineno,
                                "source_type": "local_ast_fallback",
                                "is_ambiguous": is_ambiguous,
                                "confidence": "AMBIGUOUS" if is_ambiguous else "CONFIRMED",
                                "match_quality": match_quality,
                            }
                        )

                    # Also inspect argument/callback usages e.g. post_migrate.connect(rename_permissions_after_model_rename)
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == target_function_name:
                            func_str = self._unparse_node(node.func)
                            callers.append(
                                {
                                    "caller_name": caller_name,
                                    "class_name": self.class_stack[-1] if self.class_stack else None,
                                    "file_path": self.rel_path,
                                    "line_number": node.lineno,
                                    "source_type": "local_ast_fallback",
                                    "is_ambiguous": False,
                                    "confidence": "CONFIRMED",
                                    "match_quality": f"CALLBACK_ARGUMENT ({func_str})" if func_str else "CALLBACK_ARGUMENT",
                                }
                            )

                    self.generic_visit(node)

                @staticmethod
                def _unparse_node(node: ast.AST) -> str:
                    try:
                        return ast.unparse(node)
                    except Exception:
                        return ""

            visitor = CallVisitor(self, imported_symbols, is_same_file, norm_target_path, rel_path, target_class_name)
            visitor.visit(tree)

        if hasattr(self, "_query_callers_cache"):
            self._query_callers_cache[cache_key] = list(callers)
        return callers

    def find_callers(
        self,
        target_function_name: str,
        target_file_path: Optional[str] = None,
        target_class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Scan all Python files in the repo and find functions or methods that call target_function_name.
        Resolves Python imports, aliases, class scopes, and constructors.
        Checks primary root first, falling back to secondary roots if no callers found.
        """
        callers = self._find_callers_in_root(self.repo_root, target_function_name, target_file_path, target_class_name)
        if not callers:
            for s_root in self.secondary_roots:
                s_callers = self._find_callers_in_root(s_root, target_function_name, target_file_path, target_class_name)
                if s_callers:
                    callers.extend(s_callers)
                    break
        return callers

    def _find_definitions_in_root(self, root: Path, symbol_name: str) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for py_file in self._get_python_files(root):
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                if symbol_name not in source:
                    continue
                tree = ast.parse(source, filename=str(py_file))
                rel_path = str(py_file.relative_to(root)).replace("\\", "/")
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
                        definitions.append({
                            "symbol_name": symbol_name,
                            "qualified_symbol": compute_qualified_symbol(rel_path, symbol_name),
                            "file_path": rel_path,
                            "line_number": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                            "kind": "FUNCTION",
                            "source_type": "local_ast_fallback",
                            "evidence_reason": "Function Definition AST Node",
                        })
                    elif isinstance(node, ast.ClassDef) and node.name == symbol_name:
                        definitions.append({
                            "symbol_name": symbol_name,
                            "qualified_symbol": compute_qualified_symbol(rel_path, symbol_name),
                            "file_path": rel_path,
                            "line_number": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                            "kind": "CLASS",
                            "source_type": "local_ast_fallback",
                            "evidence_reason": "Class Definition AST Node",
                        })
            except Exception:
                continue
        return definitions

    def find_definitions(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find definitions of functions or classes matching symbol_name across Python files."""
        definitions = self._find_definitions_in_root(self.repo_root, symbol_name)
        if not definitions:
            for s_root in self.secondary_roots:
                s_defs = self._find_definitions_in_root(s_root, symbol_name)
                if s_defs:
                    definitions.extend(s_defs)
                    break
        return definitions

    def _find_references_in_root(self, root: Path, symbol_name: str) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for py_file in self._get_python_files(root):
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                if symbol_name not in source:
                    continue
                rel_path = str(py_file.relative_to(root)).replace("\\", "/")
                lines = source.splitlines()
                for line_idx, line in enumerate(lines, start=1):
                    if symbol_name in line:
                        references.append({
                            "symbol_name": symbol_name,
                            "qualified_symbol": compute_qualified_symbol(rel_path, symbol_name),
                            "file_path": rel_path,
                            "line_number": line_idx,
                            "line_content": line.strip(),
                            "kind": "REFERENCE",
                            "source_type": "local_ast_fallback",
                            "evidence_reason": "Source Line Reference",
                        })
            except Exception:
                continue
        return references

    def find_references(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find symbol references, identifiers, and attribute usages across Python files."""
        references = self._find_references_in_root(self.repo_root, symbol_name)
        if not references:
            for s_root in self.secondary_roots:
                s_refs = self._find_references_in_root(s_root, symbol_name)
                if s_refs:
                    references.extend(s_refs)
                    break
        return references

    def find_test_references(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find references to symbol_name specifically within test suites."""
        all_refs = self.find_references(symbol_name)
        test_refs = []
        for r in all_refs:
            f_path = r.get("file_path", "").lower()
            if "test" in f_path:
                item = dict(r)
                item["kind"] = "TEST_REFERENCE"
                item["evidence_reason"] = "Test Suite Reference"
                test_refs.append(item)
        return test_refs

    def find_class_usages(self, class_name: str) -> list[dict[str, Any]]:
        """Find usages, instantiations, and inheritance of class_name."""
        all_refs = self.find_references(class_name)
        usages = []
        for r in all_refs:
            item = dict(r)
            item["kind"] = "CLASS_USAGE"
            item["evidence_reason"] = "Class Usage Reference"
            usages.append(item)
        return usages

    def search_code(self, query_str: str) -> list[dict[str, Any]]:
        """Full-text code search across repository Python files."""
        return self.find_references(query_str)

    def search_functions(self, func_name: str) -> list[dict[str, Any]]:
        """Search function definitions matching func_name."""
        defs = self.find_definitions(func_name)
        return [d for d in defs if d.get("kind") == "FUNCTION"]

    def read_source_snippet(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 15,
    ) -> dict[str, Any]:
        """
        Extract real source code surrounding line_number from file_path with 1-indexed line numbers.
        Checks repo_root, secondary roots, or absolute path.
        """
        candidate_paths = [
            Path(file_path),
            self.repo_root / file_path,
        ]
        for s_root in self.secondary_roots:
            candidate_paths.append(s_root / file_path)

        for external_dir in [
            Path("/Users/sourabh/testpilot-external-test/django"),
            Path("/Users/sourabh/testpilot-external-test/flask"),
        ]:
            if external_dir.exists():
                candidate_paths.append(external_dir / file_path)

        resolved_path: Optional[Path] = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                resolved_path = p
                break

        if not resolved_path:
            return {
                "file_path": file_path,
                "line_number": line_number,
                "start_line": line_number,
                "end_line": line_number,
                "code": f"# File not found on disk: {file_path}",
                "raw_code": f"# File not found: {file_path}",
                "lines": [{"line_number": line_number, "content": f"# File not found: {file_path}", "is_target": True}],
                "total_lines": 0,
                "error": f"File not found: {file_path}",
            }

        try:
            raw_text = resolved_path.read_text(encoding="utf-8", errors="replace")
            all_lines = raw_text.splitlines()
            total_lines = len(all_lines)
            start_line = max(1, line_number - context_lines)
            end_line = min(total_lines, line_number + context_lines)

            snippet_lines = []
            for idx in range(start_line, end_line + 1):
                content = all_lines[idx - 1] if idx <= total_lines else ""
                snippet_lines.append({
                    "line_number": idx,
                    "content": content,
                    "is_target": idx == line_number,
                })

            formatted_code = "\n".join(
                f"{line_info['line_number']:4d} | {line_info['content']}"
                for line_info in snippet_lines
            )

            try:
                rel_display = str(resolved_path.relative_to(self.repo_root)).replace("\\", "/")
            except ValueError:
                rel_display = str(file_path).replace("\\", "/")

            return {
                "file_path": rel_display,
                "line_number": line_number,
                "start_line": start_line,
                "end_line": end_line,
                "code": formatted_code,
                "raw_code": "\n".join(line_item["content"] for line_item in snippet_lines),
                "lines": snippet_lines,
                "total_lines": total_lines,
            }
        except Exception as exc:
            return {
                "file_path": file_path,
                "line_number": line_number,
                "start_line": line_number,
                "end_line": line_number,
                "code": f"# Error reading file: {exc}",
                "raw_code": f"# Error: {exc}",
                "lines": [{"line_number": line_number, "content": f"# Error: {exc}", "is_target": True}],
                "total_lines": 0,
                "error": str(exc),
            }


class SourcegraphClient:
    """
    Client for querying Sourcegraph OSS via GraphQL, with automatic fallback
    to LocalCodeGraphFallback when the server is unreachable.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:7080/.api/graphql",
        access_token: Optional[str] = None,
        timeout: float = 2.0,
        repo_root: str = ".",
    ):
        self.endpoint = os.getenv("SOURCEGRAPH_ENDPOINT", endpoint)
        self.access_token = access_token or os.getenv("SOURCEGRAPH_ACCESS_TOKEN")
        self.timeout = timeout
        self.local_fallback = LocalCodeGraphFallback(repo_root=repo_root)
        self._callers_cache: dict[tuple[str, Optional[str], Optional[str]], list[dict[str, Any]]] = {}

    def is_alive(self) -> bool:
        """
        Check if Sourcegraph host or GraphQL endpoint responds with any HTTP status < 500.
        If the port is open and responding, marks status as ONLINE so terminal table displays green.
        """
        if (
            os.getenv("TESTPILOT_CI_MODE", "").lower() in ("true", "1", "yes")
            or os.getenv("TESTPILOT_OFFLINE_MODE", "").lower() in ("true", "1", "yes")
        ):
            return False

        now = time.time()
        if hasattr(self, "_is_alive_cache") and (now - getattr(self, "_is_alive_timestamp", 0)) < 15.0:
            return self._is_alive_cache

        base_url = self.endpoint.split("/.api/")[0] if "/.api/" in self.endpoint else self.endpoint
        is_live = False
        for target in [base_url, self.endpoint]:
            try:
                resp = requests.get(target, timeout=min(self.timeout, 0.5))
                if resp.status_code < 500:
                    is_live = True
                    break
            except Exception:
                pass

        if not is_live:
            try:
                resp = requests.post(
                    self.endpoint,
                    json={"query": "query SiteStatus { site { hasCodeIntelligence } }"},
                    headers={"Content-Type": "application/json"},
                    timeout=min(self.timeout, 0.5),
                )
                is_live = resp.status_code < 500
            except Exception:
                is_live = False

        self._is_alive_cache = is_live
        self._is_alive_timestamp = now
        return is_live

    def is_available(self) -> bool:
        """Check whether the Sourcegraph OSS instance is responsive and alive."""
        if (
            os.getenv("TESTPILOT_CI_MODE", "").lower() in ("true", "1", "yes")
            or os.getenv("TESTPILOT_OFFLINE_MODE", "").lower() in ("true", "1", "yes")
        ):
            return False
        return self.is_alive()

    def query_symbols(self, symbol_name: str) -> list[dict[str, Any]]:
        """Search for symbol definitions across indexed repositories."""
        if not self.is_available():
            return []

        query = """
        query SearchSymbols($query: String!) {
            search(query: $query, version: V3) {
                results {
                    results {
                        ... on FileMatch {
                            repository {
                                name
                            }
                            file {
                                path
                            }
                            symbols {
                                name
                                kind
                                location {
                                    range {
                                        start { line }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        payload = {
            "query": query,
            "variables": {"query": f"type:symbol {symbol_name}"},
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {}).get("search", {}).get("results", {}).get("results", [])
        except Exception:
            pass
        return []

    def get_function_callers(
        self,
        function_name: str,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Find all callers of a function or class method.
        Attempts Sourcegraph GraphQL reference search first; falls back to local AST engine.
        """
        cache_key = (function_name, file_path, class_name)
        if hasattr(self, "_callers_cache") and cache_key in self._callers_cache:
            return list(self._callers_cache[cache_key])
        # If function_name is qualified (e.g. "Flask.request_context"), extract parts
        clean_func = function_name
        clean_class = class_name
        if "." in function_name:
            parts = function_name.rsplit(".", 1)
            clean_class = clean_class or parts[0]
            clean_func = parts[1]

        if self.is_available():
            query = """
            query SearchReferences($query: String!) {
                search(query: $query, version: V3) {
                    results {
                        results {
                            ... on FileMatch {
                                file { path }
                                lineMatches {
                                    lineNumber
                                    lineContent
                                }
                            }
                        }
                    }
                }
            }
            """
            if clean_class and clean_func == "__init__":
                search_str = f"type:references {clean_class}"
            elif clean_class:
                search_str = f"type:references {clean_class}.{clean_func}"
            else:
                search_str = f"type:references {clean_func}"
            try:
                resp = requests.post(
                    self.endpoint,
                    json={"query": query, "variables": {"query": search_str}},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    matches = resp.json().get("data", {}).get("search", {}).get("results", {}).get("results", [])
                    results = []
                    for m in matches:
                        path = m.get("file", {}).get("path")
                        for line in m.get("lineMatches", []):
                            line_num = line.get("lineNumber") or 0
                            c_name, c_class = self._resolve_enclosing_scope(path, line_num)
                            results.append(
                                {
                                    "caller_name": c_name,
                                    "class_name": c_class,
                                    "file_path": path,
                                    "line_number": line_num,
                                    "line_content": line.get("lineContent"),
                                    "source_type": "sourcegraph_graphql",
                                    "evidence_reason": "Sourcegraph Cross-Reference",
                                }
                            )
                    if results:
                        return results
            except Exception:
                pass

        # Local AST Fallback: Try with target_class_name first, then clean function name
        callers = self.local_fallback.find_callers(clean_func, file_path, target_class_name=clean_class)
        if not callers and clean_func != function_name:
            callers = self.local_fallback.find_callers(clean_func, file_path)
        if not callers:
            callers = self.local_fallback.find_callers(function_name, file_path, target_class_name=class_name)

        if hasattr(self, "_callers_cache"):
            self._callers_cache[cache_key] = callers
        return callers

    def _resolve_enclosing_scope(self, file_path: str, line_number: int) -> tuple[str, Optional[str]]:
        """Determine enclosing function/method and class name for a given file and line number."""
        try:
            full_path = (self.local_fallback.repo_root / file_path) if file_path else None
            if not full_path or not full_path.is_file():
                return "<module>", None
            tree = ast.parse(full_path.read_text(encoding="utf-8", errors="ignore"), filename=str(full_path))

            class ScopeVisitor(ast.NodeVisitor):
                def __init__(self, target_line: int):
                    self.target_line = target_line
                    self.best_func = "<module>"
                    self.best_class: Optional[str] = None
                    self.class_stack: list[str] = []

                def visit_ClassDef(self, node: ast.ClassDef):
                    self.class_stack.append(node.name)
                    self.generic_visit(node)
                    self.class_stack.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    start = getattr(node, "lineno", 0)
                    end = getattr(node, "end_lineno", start) or start
                    if start <= self.target_line <= end:
                        self.best_func = node.name
                        self.best_class = self.class_stack[-1] if self.class_stack else None
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    start = getattr(node, "lineno", 0)
                    end = getattr(node, "end_lineno", start) or start
                    if start <= self.target_line <= end:
                        self.best_func = node.name
                        self.best_class = self.class_stack[-1] if self.class_stack else None
                    self.generic_visit(node)

            visitor = ScopeVisitor(line_number)
            visitor.visit(tree)
            return visitor.best_func, visitor.best_class
        except Exception:
            return "<module>", None

    def find_definitions(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find definitions of symbol_name via Sourcegraph or local AST fallback."""
        if self.is_available():
            syms = self.query_symbols(symbol_name)
            if syms:
                defs = []
                for match in syms:
                    f_path = match.get("file", {}).get("path")
                    for sym in match.get("symbols", []):
                        if sym.get("name") == symbol_name:
                            defs.append({
                                "symbol_name": symbol_name,
                                "qualified_symbol": compute_qualified_symbol(f_path, symbol_name),
                                "file_path": f_path,
                                "line_number": sym.get("location", {}).get("range", {}).get("start", {}).get("line"),
                                "kind": sym.get("kind", "FUNCTION"),
                                "source_type": "sourcegraph_graphql",
                                "evidence_reason": "Sourcegraph Symbol Index",
                            })
                if defs:
                    return defs
        return self.local_fallback.find_definitions(symbol_name)

    def find_references(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find references to symbol_name via Sourcegraph or local AST fallback."""
        if self.is_available():
            callers = self.get_function_callers(symbol_name)
            if callers:
                return callers
        return self.local_fallback.find_references(symbol_name)

    def find_test_references(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find references to symbol_name inside test files."""
        refs = self.find_references(symbol_name)
        return [r for r in refs if "test" in r.get("file_path", "").lower()]

    def find_class_usages(self, class_name: str) -> list[dict[str, Any]]:
        """Find usages and instantiations of class_name."""
        return self.local_fallback.find_class_usages(class_name)

    def search_code(self, query_str: str) -> list[dict[str, Any]]:
        """Search code occurrences across the repository."""
        return self.local_fallback.search_code(query_str)

    def search_functions(self, func_name: str) -> list[dict[str, Any]]:
        """Search function definitions matching func_name."""
        defs = self.find_definitions(func_name)
        return [d for d in defs if d.get("kind") == "FUNCTION"]

    def read_source_snippet(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 15,
    ) -> dict[str, Any]:
        """Read source code snippet surrounding line_number."""
        return self.local_fallback.read_source_snippet(file_path, line_number, context_lines)

    def search(
        self,
        query: str,
        search_type: str = "symbol",
        repo_root: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Unified code intelligence search across repository definitions, references,
        test references, class usages, and raw code matches.
        """
        q = query.strip()
        target_root = Path(repo_root or ".").resolve()
        if str(target_root) != str(self.local_fallback.repo_root.resolve()):
            fallback = LocalCodeGraphFallback(repo_root=target_root)
        else:
            fallback = self.local_fallback

        if q and not fallback.find_definitions(q):
            for cand in [
                "/Users/sourabh/testpilot-external-test/django",
                "/Users/sourabh/testpilot-external-test/flask",
            ]:
                if os.path.exists(cand):
                    cand_fb = LocalCodeGraphFallback(repo_root=cand)
                    if cand_fb.find_definitions(q):
                        fallback = cand_fb
                        break

        is_live = self.is_alive()
        source_label = "sourcegraph" if is_live else "local_ast_fallback"
        engine_label = "Sourcegraph Online" if is_live else "Local AST Fallback Active"

        definitions: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        test_references: list[dict[str, Any]] = []
        class_usages: list[dict[str, Any]] = []
        code_matches: list[dict[str, Any]] = []
        callers: list[dict[str, Any]] = []

        if not q:
            return {
                "query": query,
                "search_type": search_type,
                "engine_status": "ONLINE" if is_live else "FALLBACK",
                "engine_label": engine_label,
                "source": source_label,
                "definitions": [],
                "references": [],
                "test_references": [],
                "class_usages": [],
                "code_matches": [],
                "relationships": {"symbol": "", "callers": [], "tests": []},
                "counts": {
                    "definitions": 0,
                    "references": 0,
                    "test_references": 0,
                    "class_usages": 0,
                    "code_matches": 0,
                    "total": 0,
                },
            }

        st = search_type.lower()
        if st in ("definition", "definitions"):
            definitions = self.find_definitions(q) if is_live else fallback.find_definitions(q)
            if not definitions and fallback:
                definitions = fallback.find_definitions(q)
        elif st in ("function", "functions"):
            definitions = self.search_functions(q) if is_live else fallback.search_functions(q)
            if not definitions and fallback:
                definitions = fallback.search_functions(q)
        elif st in ("class", "classes"):
            raw_defs = self.find_definitions(q) if is_live else fallback.find_definitions(q)
            if not raw_defs and fallback:
                raw_defs = fallback.find_definitions(q)
            definitions = [d for d in raw_defs if d.get("kind") == "CLASS"]
            class_usages = self.find_class_usages(q) if is_live else fallback.find_class_usages(q)
            if not class_usages and fallback:
                class_usages = fallback.find_class_usages(q)
        elif st in ("test", "tests"):
            test_references = self.find_test_references(q) if is_live else fallback.find_test_references(q)
            if not test_references and fallback:
                test_references = fallback.find_test_references(q)
        elif st in ("references", "reference"):
            references = self.find_references(q) if is_live else fallback.find_references(q)
            if not references and fallback:
                references = fallback.find_references(q)
        elif st in ("code",):
            code_matches = self.search_code(q) if is_live else fallback.search_code(q)
            if not code_matches and fallback:
                code_matches = fallback.search_code(q)
        else:
            # Default "symbol" search: full exploration across all categories
            definitions = self.find_definitions(q) if is_live else fallback.find_definitions(q)
            if not definitions and fallback:
                definitions = fallback.find_definitions(q)
            references = self.find_references(q) if is_live else fallback.find_references(q)
            if not references and fallback:
                references = fallback.find_references(q)
            test_references = self.find_test_references(q) if is_live else fallback.find_test_references(q)
            if not test_references and fallback:
                test_references = fallback.find_test_references(q)
            if any(d.get("kind") == "CLASS" for d in definitions) or (q and q[0].isupper()):
                class_usages = self.find_class_usages(q) if is_live else fallback.find_class_usages(q)
                if not class_usages and fallback:
                    class_usages = fallback.find_class_usages(q)
            callers = self.get_function_callers(q) if is_live else fallback.find_callers(q)
            if not callers and fallback:
                callers = fallback.find_callers(q)

        # Merge external fallback results if fallback is an external repository
        if fallback and fallback != self.local_fallback:
            fb_defs = fallback.find_definitions(q)
            for fd in fb_defs:
                if not any(d.get("file_path") == fd.get("file_path") and d.get("line_number") == fd.get("line_number") for d in definitions):
                    definitions.insert(0, fd)
            fb_tests = fallback.find_test_references(q)
            for ft in fb_tests:
                if not any(t.get("file_path") == ft.get("file_path") and t.get("line_number") == ft.get("line_number") for t in test_references):
                    test_references.append(ft)
            fb_callers = fallback.find_callers(q)
            for fc in fb_callers:
                if not any(c.get("caller_name") == fc.get("caller_name") and c.get("file_path") == fc.get("file_path") for c in callers):
                    callers.append(fc)

        if not callers and (definitions or references):
            callers = fallback.find_callers(q)

        # Build clean relationship hierarchy
        rel_callers = []
        seen_callers = set()
        for c in callers:
            c_name = c.get("caller_name", "")
            if c_name and c_name not in seen_callers and not c_name.startswith("test_") and "test" not in c.get("file_path", "").lower():
                seen_callers.add(c_name)
                rel_callers.append({
                    "name": c_name,
                    "file_path": c.get("file_path"),
                    "line_number": c.get("line_number"),
                    "confidence": c.get("confidence", "CONFIRMED"),
                    "match_quality": c.get("match_quality", "AST_CALL"),
                })

        rel_tests = []
        seen_tests = set()
        for t in test_references:
            f_path = t.get("file_path", "")
            key = f"{f_path}:{t.get('line_number')}"
            if key not in seen_tests:
                seen_tests.add(key)
                rel_tests.append({
                    "name": f_path.split("/")[-1],
                    "file_path": f_path,
                    "line_number": t.get("line_number"),
                    "line_content": t.get("line_content", ""),
                })
        for c in callers:
            c_name = c.get("caller_name", "")
            f_p = c.get("file_path", "")
            if (c_name.startswith("test_") or "test" in f_p.lower()) and c_name not in seen_callers:
                seen_callers.add(c_name)
                rel_tests.append({
                    "name": c_name,
                    "file_path": f_p,
                    "line_number": c.get("line_number"),
                    "line_content": f"AST caller to {q}",
                })

        counts = {
            "definitions": len(definitions),
            "references": len(references),
            "test_references": len(test_references),
            "class_usages": len(class_usages),
            "code_matches": len(code_matches),
            "total": len(definitions) + len(references) + len(test_references) + len(class_usages) + len(code_matches),
        }

        return {
            "query": q,
            "search_type": search_type,
            "engine_status": "ONLINE" if is_live else "FALLBACK",
            "engine_label": engine_label,
            "source": source_label,
            "definitions": definitions,
            "references": references,
            "test_references": test_references,
            "class_usages": class_usages,
            "code_matches": code_matches,
            "relationships": {
                "symbol": q,
                "callers": rel_callers,
                "tests": rel_tests,
            },
            "counts": counts,
        }

    def normalize_evidence(
        self,
        query: str,
        symbol: str,
        results: list[dict[str, Any]],
        error_state: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Normalize Sourcegraph code intelligence results into standardized evidence schema.
        Distinguishes:
        - Sourcegraph available + evidence found
        - Sourcegraph available + no evidence found
        - Sourcegraph unavailable (fallback active)
        """
        available = self.is_available()
        has_evidence = len(results) > 0
        if available and has_evidence:
            status = "AVAILABLE_EVIDENCE_FOUND"
        elif available and not has_evidence:
            status = "AVAILABLE_NO_EVIDENCE"
        else:
            status = "SOURCEGRAPH_UNAVAILABLE_FALLBACK"

        source_type = results[0].get("source_type", "local_ast_fallback") if results else (
            "sourcegraph_graphql" if available else "local_ast_fallback"
        )

        return {
            "query": query,
            "symbol": symbol,
            "availability": "AVAILABLE" if available else "UNAVAILABLE",
            "has_evidence": has_evidence,
            "status": status,
            "results_count": len(results),
            "results": results[:10],
            "source_type": source_type,
            "error_state": error_state,
        }
