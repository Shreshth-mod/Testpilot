"""
Repository Code RAG Vector Store Engine for TestPilot AI.
Persists and indexes semantic repository code units (functions, methods, classes, tests)
in a dedicated ChromaDB collection ('repo_code_store') separate from specification chunks.
Uses AST/Tree-sitter boundary extraction, qualified symbol identities, and local embeddings.
"""

import ast
import warnings
from pathlib import Path
from typing import Any, Optional

import chromadb
from pydantic import BaseModel, Field

from testpilot.rag.vector_store import LocalSentenceTransformerEmbeddingFunction

# Suppress deprecation warnings from third-party ChromaDB/opentelemetry
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CodeUnit(BaseModel):
    """Semantic code unit extracted from repository source files."""

    id: str = Field(..., description="Deterministic unique identifier")
    file_path: str = Field(..., description="Repository-relative file path")
    symbol_name: str = Field(..., description="Bare function, method, or class name")
    class_name: Optional[str] = Field(default=None, description="Enclosing class name if method")
    qualified_symbol: str = Field(..., description="Canonical qualified symbol identity e.g. Class.method")
    symbol_type: str = Field(..., description="function, method, class, or test_function")
    is_test: bool = Field(default=False, description="Whether this unit is an automated test")
    start_line: int = Field(..., description="Starting line in source file")
    end_line: int = Field(..., description="Ending line in source file")
    parameters: list[str] = Field(default_factory=list, description="Parameter names")
    docstring: str = Field(default="", description="Extracted docstring if present")
    content: str = Field(..., description="Actual source code snippet")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "class_name": self.class_name or "",
            "qualified_symbol": self.qualified_symbol,
            "symbol_type": self.symbol_type,
            "is_test": self.is_test,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parameters": ",".join(self.parameters),
            "docstring": self.docstring[:300],
        }


class RepoCodeVectorStore:
    """
    Manages persistent ChromaDB vector storage and semantic retrieval
    for repository code units in the dedicated 'repo_code_store' collection.
    """

    COLLECTION_NAME = "repo_code_store"

    EXCLUDE_DIRS = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".chroma_db",
        "htmlcov",
        "build",
        "dist",
        "site-packages",
        ".egg-info",
        "node_modules",
    }

    def __init__(self, db_path: str = ".chroma_db", repo_root: str = "."):
        self.db_path = Path(db_path)
        self.repo_root = Path(repo_root).resolve()
        self.collection_name = self.COLLECTION_NAME
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedding_fn = LocalSentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "TestPilot AI Repository Code Semantic Store"},
        )

    def count(self) -> int:
        """Return total number of code units indexed in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def extract_code_units_from_file(self, py_path: Path) -> list[CodeUnit]:
        """Parse a Python source file with AST and extract semantic code units."""
        units: list[CodeUnit] = []
        try:
            source = py_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_path))
            lines = source.splitlines()
        except Exception:
            return []

        resolved_py = py_path.resolve()
        resolved_root = self.repo_root.resolve()
        try:
            rel_path = str(resolved_py.relative_to(resolved_root)).replace("\\", "/")
        except ValueError:
            try:
                rel_path = str(py_path.relative_to(self.repo_root)).replace("\\", "/")
            except ValueError:
                rel_path = py_path.name

        is_test_file = (
            rel_path.startswith("tests/")
            or rel_path.startswith("test/")
            or py_path.name.startswith("test_")
            or py_path.name.endswith("_test.py")
        )

        class CodeUnitVisitor(ast.NodeVisitor):
            def __init__(self, outer_units: list[CodeUnit]):
                self.units = outer_units
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef):
                c_name = node.name
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l + len(node.body))
                code_snippet = "\n".join(lines[start_l - 1:end_l])
                doc = ast.get_docstring(node) or ""
                is_test_cls = is_test_file or c_name.startswith("Test")

                unit_id = f"repo_code::{rel_path}::{c_name}"
                self.units.append(
                    CodeUnit(
                        id=unit_id,
                        file_path=rel_path,
                        symbol_name=c_name,
                        class_name=None,
                        qualified_symbol=c_name,
                        symbol_type="class",
                        is_test=is_test_cls,
                        start_line=start_l,
                        end_line=end_l,
                        parameters=[],
                        docstring=doc,
                        content=code_snippet[:2000],
                    )
                )

                self.class_stack.append(c_name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._handle_func(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self._handle_func(node)

            def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
                f_name = node.name
                c_name = self.class_stack[-1] if self.class_stack else None
                qual_sym = f"{c_name}.{f_name}" if c_name else f_name
                is_test = is_test_file or f_name.startswith("test_") or f_name.endswith("_test")
                sym_type = "test_function" if is_test else ("method" if c_name else "function")

                params = [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l + len(node.body))
                code_snippet = "\n".join(lines[start_l - 1:end_l])
                doc = ast.get_docstring(node) or ""

                unit_id = f"repo_code::{rel_path}::{qual_sym}"
                self.units.append(
                    CodeUnit(
                        id=unit_id,
                        file_path=rel_path,
                        symbol_name=f_name,
                        class_name=c_name,
                        qualified_symbol=qual_sym,
                        symbol_type=sym_type,
                        is_test=is_test,
                        start_line=start_l,
                        end_line=end_l,
                        parameters=params,
                        docstring=doc,
                        content=code_snippet[:2000],
                    )
                )
                self.generic_visit(node)

        visitor = CodeUnitVisitor(units)
        visitor.visit(tree)
        return units

    def index_repository(self, target_dir: Optional[str] = None) -> int:
        """
        Scan repository Python files, extract semantic code units, and upsert
        them into the ChromaDB 'repo_code_store' collection with deduplication.
        Returns total count of indexed code units.
        """
        search_root = Path(target_dir).resolve() if target_dir else self.repo_root
        all_units: list[CodeUnit] = []

        for py_file in search_root.rglob("*.py"):
            parts = py_file.parts
            if any(p in self.EXCLUDE_DIRS or p.startswith(".") for p in parts):
                continue
            units = self.extract_code_units_from_file(py_file)
            all_units.extend(units)

        if not all_units:
            return 0

        # Deduplicate by ID
        unique_units: dict[str, CodeUnit] = {}
        for u in all_units:
            unique_units[u.id] = u

        units_to_index = list(unique_units.values())

        # Batch upsert into ChromaDB
        batch_size = 100
        for i in range(0, len(units_to_index), batch_size):
            batch = units_to_index[i:i + batch_size]
            ids = [u.id for u in batch]
            documents = [u.content for u in batch]
            metadatas = [u.metadata for u in batch]

            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception:
                pass

        return len(units_to_index)

    def retrieve_code_context(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic code retrieval from 'repo_code_store'.
        Returns list of structured results with content, qualified_symbol, file_path, metadata, distance,
        and explicit provenance (start_line, end_line, retrieval_method).
        """
        if self.count() == 0:
            # Auto-index repository if store is empty
            try:
                self.index_repository()
            except Exception:
                pass

        total = self.count()
        if total == 0:
            return []

        actual_k = min(top_k, total)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": actual_k,
        }
        if filters:
            kwargs["where"] = filters

        try:
            results = self.collection.query(**kwargs)
        except Exception:
            return []

        extracted: list[dict[str, Any]] = []
        doc_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        id_list = results.get("ids", [[]])[0]
        dist_list = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(doc_list)

        for i, doc in enumerate(doc_list):
            meta = meta_list[i] if i < len(meta_list) else {}
            u_id = id_list[i] if i < len(id_list) else ""
            dist = dist_list[i] if i < len(dist_list) else 0.0

            extracted.append({
                "source_id": u_id,
                "content": doc,
                "file_path": meta.get("file_path", ""),
                "qualified_symbol": meta.get("qualified_symbol", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_type": meta.get("symbol_type", "function"),
                "is_test": bool(meta.get("is_test", False)),
                "start_line": int(meta.get("start_line", 0)),
                "end_line": int(meta.get("end_line", 0)),
                "retrieval_method": "semantic_similarity",
                "distance": round(float(dist), 4),
                "metadata": meta,
            })

        return extracted

    def find_exact_code_unit(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Locates an exact AST code unit by symbol identity or file.
        Distinguishes exact structural provenance from semantic similarity.
        """
        # Try direct file parsing first for authoritative up-to-date syntax
        if file_path:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.repo_root / file_path
            if p.exists() and p.is_file():
                units = self.extract_code_units_from_file(p)
                for u in units:
                    if u.symbol_name == symbol_name or u.qualified_symbol == symbol_name:
                        return {
                            "source_id": u.id,
                            "content": u.content,
                            "file_path": u.file_path,
                            "qualified_symbol": u.qualified_symbol,
                            "symbol_name": u.symbol_name,
                            "symbol_type": u.symbol_type,
                            "is_test": u.is_test,
                            "start_line": u.start_line,
                            "end_line": u.end_line,
                            "retrieval_method": "exact_structural",
                            "distance": 0.0,
                            "metadata": u.metadata,
                        }

        # Fallback to vector store exact metadata query
        try:
            where_clause: dict[str, Any] = {"symbol_name": symbol_name}
            res = self.collection.get(where=where_clause, limit=1)
            if res and res.get("ids") and len(res["ids"]) > 0:
                doc = res["documents"][0] if res.get("documents") else ""
                meta = res["metadatas"][0] if res.get("metadatas") else {}
                return {
                    "source_id": res["ids"][0],
                    "content": doc,
                    "file_path": meta.get("file_path", ""),
                    "qualified_symbol": meta.get("qualified_symbol", ""),
                    "symbol_name": meta.get("symbol_name", ""),
                    "symbol_type": meta.get("symbol_type", "function"),
                    "is_test": bool(meta.get("is_test", False)),
                    "start_line": int(meta.get("start_line", 0)),
                    "end_line": int(meta.get("end_line", 0)),
                    "retrieval_method": "exact_structural",
                    "distance": 0.0,
                    "metadata": meta,
                }
        except Exception:
            pass

        return None

    def retrieve_symbol_evidence(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
        include_callers: bool = True,
        include_tests: bool = True,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Targeted multi-tier retrieval for Repository Evolution:
        1. Exact structural definition of the symbol (exact_structural)
        2. Semantic similarity search across related methods/contracts (semantic_similarity)
        3. Relevant automated tests and assertions (semantic_test_evidence)
        Preserves complete provenance, deduplicates, and marks retrieval methods clearly.
        """
        evidence_results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Tier 1: Exact structural definition
        exact = self.find_exact_code_unit(symbol_name=symbol_name, file_path=file_path)
        if exact:
            seen_ids.add(exact["source_id"])
            evidence_results.append(exact)

        # Tier 2: Semantic code context & callers
        if include_callers:
            query = f"{symbol_name} caller invocation contract dataflow {file_path or ''}"
            contexts = self.retrieve_code_context(query=query, top_k=top_k)
            for c in contexts:
                s_id = c.get("source_id", "")
                if s_id and s_id not in seen_ids:
                    seen_ids.add(s_id)
                    evidence_results.append(c)

        # Tier 3: Associated test units exercising this contract
        if include_tests:
            test_query = f"test_{symbol_name} assert {symbol_name} fixture"
            try:
                test_results = self.retrieve_code_context(
                    query=test_query,
                    top_k=min(top_k, 4),
                    filters={"is_test": True},
                )
                for t in test_results:
                    s_id = t.get("source_id", "")
                    if s_id and s_id not in seen_ids:
                        seen_ids.add(s_id)
                        t["retrieval_method"] = "semantic_test_evidence"
                        evidence_results.append(t)
            except Exception:
                pass

        return evidence_results
