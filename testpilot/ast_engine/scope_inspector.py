"""
Target Scope Inspector for TestPilot AI Test Studio.
Resolves arbitrary target scopes (entire repository, folders, files, classes,
functions, manual symbols, or natural language feature descriptions) into concrete
AST symbols, source files, and branch specifications.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Optional
from pydantic import BaseModel, Field

from testpilot.ast_engine.treesitter_parser import ASTDiffParser
from testpilot.core.models import ASTFunctionDef

logger = logging.getLogger("testpilot.ast_engine.scope_inspector")


class DiscoveredSymbol(BaseModel):
    """A concrete function, method, or class discovered within the inspected scope."""
    name: str
    symbol_type: str  # "function", "method", "class"
    file_path: str
    relative_path: str
    class_name: Optional[str] = None
    line_number: int = 1
    docstring: Optional[str] = None
    parameters: list[str] = Field(default_factory=list)
    branch_count: int = 0
    relevance_score: float = 1.0


class ScopeInspectionResult(BaseModel):
    """Authoritative result of inspecting a user-specified target scope."""
    scope_type: str  # "entire_repo", "folder", "file", "class", "function", "feature", "symbols"
    target_query: str
    repo_path: str
    matched_files: list[str] = Field(default_factory=list)
    symbols: list[DiscoveredSymbol] = Field(default_factory=list)
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    feature_rationale: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class ScopeInspector:
    """Introspects and resolves target scopes across codebases."""

    EXCLUDED_DIRS = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        "tests", "test", "testing", ".mypy_cache", ".tox", "build", "dist",
        "node_modules", ".idea", ".vscode"
    }

    EXCLUDED_FILES = {
        "setup.py", "conftest.py", "test_*.py", "*_test.py"
    }

    @classmethod
    def inspect(
        cls,
        repo_path: Path | str,
        scope_type: str = "entire_repo",
        target_query: str = "",
        max_files: int = 100,
    ) -> ScopeInspectionResult:
        """
        Inspects the codebase according to scope_type:
        - 'entire_repo': All non-test Python files in the repository.
        - 'folder': All Python files under the specified folder.
        - 'file': A specific file path.
        - 'class': A specific class name.
        - 'function': A specific function or method name.
        - 'feature': Semantic & keyword feature search.
        - 'symbols': Comma-separated list of symbols.
        """
        base_dir = Path(repo_path).expanduser().resolve()
        stype = (scope_type or "entire_repo").lower().strip()
        query = (target_query or "").strip()

        if stype in ("entire_repo", "repo", "all"):
            return cls._inspect_entire_repo(base_dir, max_files=max_files)
        elif stype in ("commits", "commit", "diff"):
            return cls._inspect_commits(base_dir, commit_range=query, max_files=max_files)
        elif stype == "folder":
            return cls._inspect_folder(base_dir, folder_path=query, max_files=max_files)
        elif stype == "file":
            return cls._inspect_file(base_dir, file_path=query)
        elif stype == "class":
            return cls._inspect_class(base_dir, class_name=query)
        elif stype in ("function", "method"):
            return cls._inspect_function(base_dir, func_name=query)
        elif stype in ("feature", "feature_name", "prompt"):
            return cls._inspect_feature(base_dir, feature_name=query, max_files=max_files)
        elif stype in ("symbols", "symbols_list"):
            return cls._inspect_symbols(base_dir, symbols_query=query)
        else:
            return cls._inspect_entire_repo(base_dir, max_files=max_files)

    @classmethod
    def _collect_py_files(cls, base_dir: Path, subfolder: Optional[Path] = None) -> list[Path]:
        """Collects clean source Python files, strictly excluding test files and vendor dirs."""
        root = subfolder if (subfolder and subfolder.is_dir()) else base_dir
        clean_files: list[Path] = []

        for p in root.rglob("*.py"):
            parts = p.parts
            if any(ex in parts for ex in cls.EXCLUDED_DIRS):
                continue
            name = p.name
            if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
                continue
            if name == "setup.py":
                continue
            clean_files.append(p)

        return sorted(clean_files)

    @classmethod
    def _inspect_entire_repo(cls, base_dir: Path, max_files: int = 100) -> ScopeInspectionResult:
        """Inspects all testable source files across the entire repository."""
        py_files = cls._collect_py_files(base_dir)[:max_files]
        matched_rel_files: list[str] = []
        symbols: list[DiscoveredSymbol] = []
        classes_found: set[str] = set()

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                if not funcs:
                    continue
                matched_rel_files.append(rel_p)
                for fn in funcs:
                    if fn.class_name:
                        classes_found.add(f"{rel_p}::{fn.class_name}")
                    symbols.append(
                        DiscoveredSymbol(
                            name=fn.name,
                            symbol_type="method" if fn.class_name else "function",
                            file_path=str(f),
                            relative_path=rel_p,
                            class_name=fn.class_name,
                            line_number=fn.line_start,
                            docstring=fn.docstring,
                            parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                            branch_count=len(getattr(fn, "branch_conditions", [])),
                            relevance_score=1.0,
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not parse file {f}: {e}")

        return ScopeInspectionResult(
            scope_type="entire_repo",
            target_query="Entire Repository",
            repo_path=str(base_dir),
            matched_files=matched_rel_files,
            symbols=symbols,
            total_files=len(matched_rel_files),
            total_functions=len(symbols),
            total_classes=len(classes_found),
        )

    @classmethod
    def _inspect_folder(cls, base_dir: Path, folder_path: str, max_files: int = 100) -> ScopeInspectionResult:
        """Inspects all Python files inside a specific folder."""
        folder = (base_dir / folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            folder = base_dir

        py_files = cls._collect_py_files(base_dir, subfolder=folder)[:max_files]
        matched_rel_files: list[str] = []
        symbols: list[DiscoveredSymbol] = []
        classes_found: set[str] = set()

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                if not funcs:
                    continue
                matched_rel_files.append(rel_p)
                for fn in funcs:
                    if fn.class_name:
                        classes_found.add(f"{rel_p}::{fn.class_name}")
                    symbols.append(
                        DiscoveredSymbol(
                            name=fn.name,
                            symbol_type="method" if fn.class_name else "function",
                            file_path=str(f),
                            relative_path=rel_p,
                            class_name=fn.class_name,
                            line_number=fn.line_start,
                            docstring=fn.docstring,
                            parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                            branch_count=len(getattr(fn, "branch_conditions", [])),
                            relevance_score=1.0,
                        )
                    )
            except Exception:
                pass

        return ScopeInspectionResult(
            scope_type="folder",
            target_query=str(folder.relative_to(base_dir)).replace("\\", "/"),
            repo_path=str(base_dir),
            matched_files=matched_rel_files,
            symbols=symbols,
            total_files=len(matched_rel_files),
            total_functions=len(symbols),
            total_classes=len(classes_found),
        )

    @classmethod
    def _inspect_file(cls, base_dir: Path, file_path: str) -> ScopeInspectionResult:
        """Inspects an individual source file."""
        target = Path(file_path)
        if not target.is_absolute():
            target = base_dir / file_path
        target = target.resolve()

        if not target.exists() or not target.is_file():
            # Try finding by basename in repo
            matches = list(base_dir.rglob(target.name))
            clean_matches = [m for m in matches if not any(ex in m.parts for ex in cls.EXCLUDED_DIRS)]
            if clean_matches:
                target = clean_matches[0]
            else:
                return ScopeInspectionResult(
                    scope_type="file",
                    target_query=file_path,
                    repo_path=str(base_dir),
                    warnings=[f"File '{file_path}' not found in repository."],
                )

        rel_p = str(target.relative_to(base_dir)).replace("\\", "/")
        funcs = ASTDiffParser.parse_file(str(target))
        classes_found = {f"{rel_p}::{fn.class_name}" for fn in funcs if fn.class_name}

        symbols = [
            DiscoveredSymbol(
                name=fn.name,
                symbol_type="method" if fn.class_name else "function",
                file_path=str(target),
                relative_path=rel_p,
                class_name=fn.class_name,
                line_number=fn.line_start,
                docstring=fn.docstring,
                parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                branch_count=len(getattr(fn, "branch_conditions", [])),
                relevance_score=1.0,
            )
            for fn in funcs
        ]

        return ScopeInspectionResult(
            scope_type="file",
            target_query=rel_p,
            repo_path=str(base_dir),
            matched_files=[rel_p],
            symbols=symbols,
            total_files=1,
            total_functions=len(symbols),
            total_classes=len(classes_found),
        )

    @classmethod
    def _inspect_class(cls, base_dir: Path, class_name: str) -> ScopeInspectionResult:
        """Inspects all methods and state of a target class."""
        clean_class = class_name.split("::")[-1].strip()
        py_files = cls._collect_py_files(base_dir)
        matched_files: list[str] = []
        symbols: list[DiscoveredSymbol] = []

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                class_methods = [fn for fn in funcs if fn.class_name and fn.class_name.lower() == clean_class.lower()]
                if class_methods:
                    matched_files.append(rel_p)
                    for fn in class_methods:
                        symbols.append(
                            DiscoveredSymbol(
                                name=fn.name,
                                symbol_type="method",
                                file_path=str(f),
                                relative_path=rel_p,
                                class_name=fn.class_name,
                                line_number=fn.line_start,
                                docstring=fn.docstring,
                                parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                                branch_count=len(getattr(fn, "branch_conditions", [])),
                                relevance_score=1.0,
                            )
                        )
            except Exception:
                pass

        return ScopeInspectionResult(
            scope_type="class",
            target_query=clean_class,
            repo_path=str(base_dir),
            matched_files=matched_files,
            symbols=symbols,
            total_files=len(matched_files),
            total_functions=len(symbols),
            total_classes=1 if symbols else 0,
            warnings=[] if symbols else [f"Class '{clean_class}' not found in repository."],
        )

    @classmethod
    def _inspect_function(cls, base_dir: Path, func_name: str) -> ScopeInspectionResult:
        """Inspects an individual function or method."""
        clean_fn = func_name.split("::")[-1].strip()
        py_files = cls._collect_py_files(base_dir)
        matched_files: list[str] = []
        symbols: list[DiscoveredSymbol] = []

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                matching = [fn for fn in funcs if fn.name.lower() == clean_fn.lower()]
                if matching:
                    matched_files.append(rel_p)
                    for fn in matching:
                        symbols.append(
                            DiscoveredSymbol(
                                name=fn.name,
                                symbol_type="method" if fn.class_name else "function",
                                file_path=str(f),
                                relative_path=rel_p,
                                class_name=fn.class_name,
                                line_number=fn.line_start,
                                docstring=fn.docstring,
                                parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                                branch_count=len(getattr(fn, "branch_conditions", [])),
                                relevance_score=1.0,
                            )
                        )
            except Exception:
                pass

        return ScopeInspectionResult(
            scope_type="function",
            target_query=clean_fn,
            repo_path=str(base_dir),
            matched_files=matched_files,
            symbols=symbols,
            total_files=len(matched_files),
            total_functions=len(symbols),
            total_classes=len({s.class_name for s in symbols if s.class_name}),
            warnings=[] if symbols else [f"Function '{clean_fn}' not found in repository."],
        )

    @classmethod
    def _inspect_symbols(cls, base_dir: Path, symbols_query: str) -> ScopeInspectionResult:
        """Inspects a set of comma- or space-separated symbols."""
        sym_names = [s.strip() for s in re.split(r"[,;|\s]+", symbols_query) if s.strip()]
        sym_set = {s.lower() for s in sym_names}

        py_files = cls._collect_py_files(base_dir)
        matched_files: set[str] = set()
        symbols: list[DiscoveredSymbol] = []

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                for fn in funcs:
                    if fn.name.lower() in sym_set or (fn.class_name and fn.class_name.lower() in sym_set):
                        matched_files.add(rel_p)
                        symbols.append(
                            DiscoveredSymbol(
                                name=fn.name,
                                symbol_type="method" if fn.class_name else "function",
                                file_path=str(f),
                                relative_path=rel_p,
                                class_name=fn.class_name,
                                line_number=fn.line_start,
                                docstring=fn.docstring,
                                parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                                branch_count=len(getattr(fn, "branch_conditions", [])),
                                relevance_score=1.0,
                            )
                        )
            except Exception:
                pass

        return ScopeInspectionResult(
            scope_type="symbols",
            target_query=symbols_query,
            repo_path=str(base_dir),
            matched_files=sorted(list(matched_files)),
            symbols=symbols,
            total_files=len(matched_files),
            total_functions=len(symbols),
            total_classes=len({s.class_name for s in symbols if s.class_name}),
        )

    @classmethod
    def _inspect_feature(
        cls,
        base_dir: Path,
        feature_name: str,
        max_files: int = 15,
    ) -> ScopeInspectionResult:
        """
        Multi-layer hybrid resolution for a natural language feature name:
        1. Git commit log mining (searching commit messages for the feature).
        2. Lexical & AST docstring/symbol matching.
        """
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]{3,}", feature_name)]
        if not tokens:
            tokens = [feature_name.lower().strip()]

        file_scores: dict[str, float] = {}
        matched_symbols: list[DiscoveredSymbol] = []
        git_matched_files: list[str] = []

        # Layer 1: Git commit history mining if repo has .git
        if (base_dir / ".git").exists():
            for tok in tokens[:3]:
                try:
                    res = subprocess.run(
                        ["git", "log", f"--grep={tok}", "-n", "5", "--name-only", "--pretty=format:"],
                        cwd=str(base_dir),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        for line in res.stdout.splitlines():
                            clean_line = line.strip().replace("\\", "/")
                            if clean_line.endswith(".py") and not any(ex in clean_line for ex in cls.EXCLUDED_DIRS):
                                file_scores[clean_line] = file_scores.get(clean_line, 0.0) + 2.5
                                if clean_line not in git_matched_files:
                                    git_matched_files.append(clean_line)
                except Exception:
                    pass

        # Layer 2: AST Docstring, Class, and Function Name matching
        py_files = cls._collect_py_files(base_dir)

        for f in py_files:
            rel_p = str(f.relative_to(base_dir)).replace("\\", "/")
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                file_hit = False

                for fn in funcs:
                    fn_name_lower = fn.name.lower()
                    doc_lower = (fn.docstring or "").lower()
                    cls_lower = (fn.class_name or "").lower()

                    match_score = 0.0
                    for tok in tokens:
                        if tok in fn_name_lower:
                            match_score += 3.0
                        if tok in cls_lower:
                            match_score += 2.0
                        if tok in doc_lower:
                            match_score += 1.0

                    if match_score > 0.0:
                        file_scores[rel_p] = file_scores.get(rel_p, 0.0) + match_score
                        matched_symbols.append(
                            DiscoveredSymbol(
                                name=fn.name,
                                symbol_type="method" if fn.class_name else "function",
                                file_path=str(f),
                                relative_path=rel_p,
                                class_name=fn.class_name,
                                line_number=fn.line_start,
                                docstring=fn.docstring,
                                parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                                branch_count=len(getattr(fn, "branch_conditions", [])),
                                relevance_score=round(match_score, 2),
                            )
                        )
                        file_hit = True

                if not file_hit:
                    # Check filename itself
                    for tok in tokens:
                        if tok in f.name.lower():
                            file_scores[rel_p] = file_scores.get(rel_p, 0.0) + 1.5
            except Exception:
                pass

        # Sort files by relevance
        sorted_files = sorted(file_scores.keys(), key=lambda k: file_scores[k], reverse=True)[:max_files]
        sorted_symbols = sorted(matched_symbols, key=lambda s: s.relevance_score, reverse=True)[:50]

        rationale = (
            f"Feature '{feature_name}' resolved across {len(sorted_files)} source files "
            f"via AST docstrings/symbols ({len(sorted_symbols)} matching functions)"
            + (f" and {len(git_matched_files)} historical git commits." if git_matched_files else ".")
        )

        return ScopeInspectionResult(
            scope_type="feature",
            target_query=feature_name,
            repo_path=str(base_dir),
            matched_files=sorted_files,
            symbols=sorted_symbols,
            total_files=len(sorted_files),
            total_functions=len(sorted_symbols),
            total_classes=len({s.class_name for s in sorted_symbols if s.class_name}),
            feature_rationale=rationale,
            warnings=[] if sorted_files else [f"No code files matched feature query '{feature_name}'."],
        )

    @classmethod
    def _inspect_commits(
        cls,
        base_dir: Path,
        commit_range: str = "",
        max_files: int = 100,
    ) -> ScopeInspectionResult:
        """Inspects files changed in a Git commit range (e.g. HEAD~1..HEAD or commit SHA)."""
        ref = commit_range.strip() or "HEAD~1..HEAD"
        matched_rel_files: list[str] = []
        symbols: list[DiscoveredSymbol] = []
        classes_found: set[str] = set()
        warnings: list[str] = []

        try:
            cmd = ["git", "diff", "--name-only", ref]
            if ".." not in ref:
                cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", ref]
            res = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.splitlines():
                    rel_p = line.strip().replace("\\", "/")
                    if rel_p.endswith(".py") and not any(ex in rel_p for ex in cls.EXCLUDED_DIRS):
                        f = base_dir / rel_p
                        if f.exists() and f.is_file():
                            matched_rel_files.append(rel_p)
            else:
                warnings.append(f"No changed Python files found in commit range '{ref}'.")
        except Exception as e:
            warnings.append(f"Git diff error: {e}")

        # If no changed files in diff, fallback to recent git status or modified files
        if not matched_rel_files:
            try:
                st = subprocess.run(["git", "status", "--porcelain"], cwd=str(base_dir), capture_output=True, text=True, check=False)
                if st.returncode == 0:
                    for line in st.stdout.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            rel_p = parts[-1].replace("\\", "/")
                            if rel_p.endswith(".py") and not any(ex in rel_p for ex in cls.EXCLUDED_DIRS):
                                f = base_dir / rel_p
                                if f.exists() and f.is_file() and rel_p not in matched_rel_files:
                                    matched_rel_files.append(rel_p)
            except Exception:
                pass

        for rel_p in matched_rel_files[:max_files]:
            f = base_dir / rel_p
            try:
                funcs = ASTDiffParser.parse_file(str(f))
                for fn in funcs:
                    if fn.class_name:
                        classes_found.add(f"{rel_p}::{fn.class_name}")
                    symbols.append(
                        DiscoveredSymbol(
                            name=fn.name,
                            symbol_type="method" if fn.class_name else "function",
                            file_path=str(f),
                            relative_path=rel_p,
                            class_name=fn.class_name,
                            line_number=fn.line_start,
                            docstring=fn.docstring,
                            parameters=[p.name for p in fn.parameters if p.name not in ("self", "cls")],
                            branch_count=len(getattr(fn, "branch_conditions", [])),
                            relevance_score=1.0,
                        )
                    )
            except Exception:
                pass

        return ScopeInspectionResult(
            scope_type="commits",
            target_query=ref,
            repo_path=str(base_dir),
            matched_files=matched_rel_files,
            symbols=symbols,
            total_files=len(matched_rel_files),
            total_functions=len(symbols),
            total_classes=len(classes_found),
            warnings=warnings,
        )
