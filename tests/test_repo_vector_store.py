"""
Unit tests for RepoCodeVectorStore:
Validates semantic code unit extraction, deterministic ID generation,
idempotent deduplication, metadata completeness, and semantic retrieval.
"""

import tempfile
from pathlib import Path

import pytest

from testpilot.rag.repo_vector_store import RepoCodeVectorStore


@pytest.fixture
def temp_repo_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        root = Path(tmpdir)
        service_file = root / "service.py"
        service_file.write_text(
            '''"""Order processing service."""

class OrderProcessor:
    """Manages order lifecycle."""

    def __init__(self, tax_rate: float = 0.08):
        self.tax_rate = tax_rate

    def calculate_total(self, subtotal: float) -> float:
        """Calculate total with tax."""
        return subtotal * (1.0 + self.tax_rate)


def standalone_helper(x: int) -> int:
    """Standalone utility function."""
    return x * 2
''',
            encoding="utf-8",
        )

        test_file = root / "test_service.py"
        test_file.write_text(
            '''"""Tests for OrderProcessor."""
from service import OrderProcessor

def test_calculate_total():
    proc = OrderProcessor()
    assert proc.calculate_total(100.0) == 108.0
''',
            encoding="utf-8",
        )

        yield root


def test_extract_code_units(temp_repo_dir):
    store = RepoCodeVectorStore(db_path=str(temp_repo_dir / ".chroma_test"), repo_root=str(temp_repo_dir))
    units = store.extract_code_units_from_file(temp_repo_dir / "service.py")

    assert len(units) >= 3
    names = {u.symbol_name for u in units}
    assert "OrderProcessor" in names
    assert "calculate_total" in names
    assert "standalone_helper" in names

    # Verify qualified symbol
    method_unit = next(u for u in units if u.symbol_name == "calculate_total")
    assert method_unit.qualified_symbol == "OrderProcessor.calculate_total"
    assert method_unit.symbol_type == "method"
    assert method_unit.is_test is False
    assert method_unit.docstring == "Calculate total with tax."
    assert "subtotal" in method_unit.parameters

    # Verify deterministic ID format
    assert method_unit.id == "repo_code::service.py::OrderProcessor.calculate_total"


def test_extract_test_units(temp_repo_dir):
    store = RepoCodeVectorStore(db_path=str(temp_repo_dir / ".chroma_test"), repo_root=str(temp_repo_dir))
    test_units = store.extract_code_units_from_file(temp_repo_dir / "test_service.py")

    assert len(test_units) >= 1
    t_unit = test_units[0]
    assert t_unit.symbol_name == "test_calculate_total"
    assert t_unit.is_test is True
    assert t_unit.symbol_type == "test_function"
    assert t_unit.id == "repo_code::test_service.py::test_calculate_total"


def test_index_and_retrieve_repository(temp_repo_dir):
    db_path = str(temp_repo_dir / ".chroma_test")
    store = RepoCodeVectorStore(db_path=db_path, repo_root=str(temp_repo_dir))

    count = store.index_repository()
    assert count >= 3
    assert store.count() >= 3

    # Retrieval
    results = store.retrieve_code_context(query="calculate tax total", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    top_result = results[0]
    assert "content" in top_result
    assert "qualified_symbol" in top_result
    assert "file_path" in top_result
    assert "metadata" in top_result


def test_deduplication_and_idempotency(temp_repo_dir):
    db_path = str(temp_repo_dir / ".chroma_test")
    store = RepoCodeVectorStore(db_path=db_path, repo_root=str(temp_repo_dir))

    count_first = store.index_repository()
    count_second = store.index_repository()

    assert count_first == count_second
    # Total count in collection shouldn't double because of deterministic upsert
    assert store.count() == count_first
