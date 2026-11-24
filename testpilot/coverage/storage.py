"""
SQLite and filesystem persistence store for TestPilot Test Coverage Intelligence.
Maintains indexed coverage runs, file coverage metrics, function breakdowns,
and test-to-code mapping records across repositories and revisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional

from testpilot.coverage.models import (
    CoverageEvidenceType,
    CoverageReport,
    CoverageStatus,
    FileCoverageSummary,
    FunctionCoverage,
    FunctionStatus,
    LineCoverageData,
    RepositoryCoverageSummary,
    TestToCodeMapping,
)

logger = logging.getLogger("testpilot.coverage.storage")


class CoverageStore:
    """
    Thread-safe, indexed SQLite persistence store for Test Coverage reports and metadata.
    Isolates coverage records strictly by repo_id, commit_hash, workflow_id, and execution_run_id.
    """

    _instance: Optional[CoverageStore] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path | str] = None, reports_dir: Optional[Path | str] = None) -> None:
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path(".testpilot/coverage.db").resolve()

        if reports_dir:
            self.reports_dir = Path(reports_dir).resolve()
        else:
            self.reports_dir = Path(".testpilot/reports/coverage").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get_instance(
        cls,
        db_path: Optional[Path | str] = None,
        reports_dir: Optional[Path | str] = None,
    ) -> CoverageStore:
        """Singleton accessor for global CoverageStore."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path, reports_dir)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for testing)."""
        with cls._lock:
            cls._instance = None

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes for high-speed coverage querying."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_runs (
                    coverage_run_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    execution_run_id TEXT,
                    repo_id TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    is_full_suite INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    total_statements INTEGER DEFAULT 0,
                    covered_statements INTEGER DEFAULT 0,
                    missing_statements INTEGER DEFAULT 0,
                    line_coverage_pct REAL DEFAULT 0.0,
                    total_functions INTEGER DEFAULT 0,
                    covered_functions INTEGER DEFAULT 0,
                    partially_covered_functions INTEGER DEFAULT 0,
                    uncovered_functions INTEGER DEFAULT 0,
                    function_coverage_pct REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    report_json_path TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coverage_run_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    total_statements INTEGER DEFAULT 0,
                    covered_statements INTEGER DEFAULT 0,
                    missing_statements INTEGER DEFAULT 0,
                    line_coverage_pct REAL DEFAULT 0.0,
                    executed_lines_json TEXT,
                    missing_lines_json TEXT,
                    executable_lines_json TEXT,
                    FOREIGN KEY(coverage_run_id) REFERENCES coverage_runs(coverage_run_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS function_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coverage_run_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    class_name TEXT,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    executable_lines_json TEXT,
                    executed_lines_json TEXT,
                    missing_lines_json TEXT,
                    coverage_pct REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    is_nested INTEGER DEFAULT 0,
                    executing_tests_json TEXT,
                    FOREIGN KEY(coverage_run_id) REFERENCES coverage_runs(coverage_run_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_code_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coverage_run_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    covered_files_json TEXT,
                    covered_functions_json TEXT,
                    executed_lines_json TEXT,
                    evidence_type TEXT NOT NULL,
                    FOREIGN KEY(coverage_run_id) REFERENCES coverage_runs(coverage_run_id) ON DELETE CASCADE
                );
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cov_repo_commit ON coverage_runs(repo_id, commit_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cov_workflow ON coverage_runs(workflow_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_cov_run ON file_coverage(coverage_run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_func_cov_run ON function_coverage(coverage_run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_func_cov_qual ON function_coverage(repo_id, qualified_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_map_run ON test_code_mappings(coverage_run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_map_id ON test_code_mappings(repo_id, test_id);")
            conn.commit()

    def save_report(self, report: CoverageReport) -> Path:
        """
        Persists a complete coverage report both to SQLite tables for structured querying
        and to a JSON artifact on disk for complete object reconstruction.
        """
        json_path = self.reports_dir / f"{report.coverage_run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        with self._get_connection() as conn:
            # 1. Insert coverage_runs
            conn.execute(
                """
                INSERT OR REPLACE INTO coverage_runs (
                    coverage_run_id, workflow_id, execution_run_id, repo_id, repo_path,
                    commit_hash, execution_mode, is_full_suite, status, error_message,
                    total_statements, covered_statements, missing_statements, line_coverage_pct,
                    total_functions, covered_functions, partially_covered_functions,
                    uncovered_functions, function_coverage_pct, created_at, report_json_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.coverage_run_id,
                    report.workflow_id,
                    report.execution_run_id,
                    report.repo_id,
                    report.repo_path,
                    report.commit_hash,
                    report.execution_mode,
                    1 if report.is_full_suite else 0,
                    report.status.value,
                    report.error_message,
                    report.total_statements,
                    report.covered_statements,
                    report.missing_statements,
                    report.line_coverage_pct,
                    report.total_functions,
                    report.covered_functions,
                    report.partially_covered_functions,
                    report.uncovered_functions,
                    report.function_coverage_pct,
                    report.created_at,
                    str(json_path),
                ),
            )

            # 2. Insert file_coverage
            conn.execute("DELETE FROM file_coverage WHERE coverage_run_id = ?", (report.coverage_run_id,))
            file_rows = []
            for file_path, fcov in report.files.items():
                file_rows.append((
                    report.coverage_run_id,
                    report.repo_id,
                    report.commit_hash,
                    file_path,
                    fcov.total_statements,
                    fcov.covered_statements,
                    len(fcov.missing_lines),
                    fcov.line_coverage_pct,
                    json.dumps(fcov.executed_lines),
                    json.dumps(fcov.missing_lines),
                    json.dumps(fcov.executable_lines),
                ))
            if file_rows:
                conn.executemany(
                    """
                    INSERT INTO file_coverage (
                        coverage_run_id, repo_id, commit_hash, file_path,
                        total_statements, covered_statements, missing_statements,
                        line_coverage_pct, executed_lines_json, missing_lines_json, executable_lines_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    file_rows,
                )

            # 3. Insert function_coverage
            conn.execute("DELETE FROM function_coverage WHERE coverage_run_id = ?", (report.coverage_run_id,))
            func_rows = []
            for fsum in report.file_summaries:
                for fn in fsum.functions:
                    func_rows.append((
                        report.coverage_run_id,
                        report.repo_id,
                        report.commit_hash,
                        fn.file_path,
                        fn.function_name,
                        fn.qualified_name,
                        fn.class_name,
                        fn.start_line,
                        fn.end_line,
                        json.dumps(fn.executable_lines),
                        json.dumps(fn.executed_lines),
                        json.dumps(fn.missing_lines),
                        fn.coverage_pct,
                        fn.status.value,
                        1 if fn.is_nested else 0,
                        json.dumps(fn.executing_tests),
                    ))
            if func_rows:
                conn.executemany(
                    """
                    INSERT INTO function_coverage (
                        coverage_run_id, repo_id, commit_hash, file_path,
                        function_name, qualified_name, class_name, start_line, end_line,
                        executable_lines_json, executed_lines_json, missing_lines_json,
                        coverage_pct, status, is_nested, executing_tests_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    func_rows,
                )

            # 4. Insert test_code_mappings
            conn.execute("DELETE FROM test_code_mappings WHERE coverage_run_id = ?", (report.coverage_run_id,))
            map_rows = []
            for tm in report.test_mappings:
                map_rows.append((
                    report.coverage_run_id,
                    report.repo_id,
                    report.commit_hash,
                    tm.test_id,
                    tm.test_name,
                    tm.test_file,
                    json.dumps(tm.covered_files),
                    json.dumps(tm.covered_functions),
                    json.dumps(tm.executed_lines_by_file),
                    tm.evidence_type.value,
                ))
            if map_rows:
                conn.executemany(
                    """
                    INSERT INTO test_code_mappings (
                        coverage_run_id, repo_id, commit_hash, test_id, test_name,
                        test_file, covered_files_json, covered_functions_json,
                        executed_lines_json, evidence_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    map_rows,
                )

            conn.commit()

        logger.info(f"Saved coverage report '{report.coverage_run_id}' for repo '{report.repo_id}' at commit '{report.commit_hash}'.")
        return json_path

    def get_report(self, coverage_run_id: str) -> Optional[CoverageReport]:
        """Loads a coverage report by ID from disk cache or SQLite."""
        json_path = self.reports_dir / f"{coverage_run_id}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return CoverageReport.model_validate_json(f.read())
            except Exception as e:
                logger.warning(f"Could not load coverage json {json_path}: {e}")

        # Fallback to query from SQLite
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM coverage_runs WHERE coverage_run_id = ?",
                (coverage_run_id,),
            ).fetchone()
            if not row:
                return None

            files_dict: dict[str, LineCoverageData] = {}
            frows = conn.execute(
                "SELECT * FROM file_coverage WHERE coverage_run_id = ?",
                (coverage_run_id,),
            ).fetchall()
            for fr in frows:
                files_dict[fr["file_path"]] = LineCoverageData(
                    file_path=fr["file_path"],
                    total_statements=fr["total_statements"],
                    covered_statements=fr["covered_statements"],
                    missing_lines=json.loads(fr["missing_lines_json"] or "[]"),
                    executed_lines=json.loads(fr["executed_lines_json"] or "[]"),
                    executable_lines=json.loads(fr["executable_lines_json"] or "[]"),
                    line_coverage_pct=fr["line_coverage_pct"],
                )

            return CoverageReport(
                coverage_run_id=row["coverage_run_id"],
                workflow_id=row["workflow_id"],
                execution_run_id=row["execution_run_id"],
                repo_id=row["repo_id"],
                repo_path=row["repo_path"],
                commit_hash=row["commit_hash"],
                execution_mode=row["execution_mode"],
                is_full_suite=bool(row["is_full_suite"]),
                status=CoverageStatus(row["status"]),
                error_message=row["error_message"],
                total_statements=row["total_statements"],
                covered_statements=row["covered_statements"],
                missing_statements=row["missing_statements"],
                line_coverage_pct=row["line_coverage_pct"],
                total_functions=row["total_functions"],
                covered_functions=row["covered_functions"],
                partially_covered_functions=row["partially_covered_functions"],
                uncovered_functions=row["uncovered_functions"],
                function_coverage_pct=row["function_coverage_pct"],
                files=files_dict,
                created_at=row["created_at"],
            )

    def get_latest_report(
        self,
        repo_id: str,
        commit_hash: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Optional[CoverageReport]:
        """Fetches the latest coverage report matching repo_id and optional commit/workflow filters."""
        query = "SELECT coverage_run_id FROM coverage_runs WHERE repo_id = ?"
        params: list[Any] = [repo_id]

        if commit_hash:
            query += " AND commit_hash = ?"
            params.append(commit_hash)
        if workflow_id:
            query += " AND workflow_id = ?"
            params.append(workflow_id)

        query += " ORDER BY created_at DESC LIMIT 1"

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if not row:
                return None
            return self.get_report(row["coverage_run_id"])

    def list_reports(
        self,
        repo_id: Optional[str] = None,
        commit_hash: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Lists historical coverage runs with summary metrics."""
        query = "SELECT * FROM coverage_runs"
        params: list[Any] = []
        conditions: list[str] = []

        if repo_id:
            conditions.append("repo_id = ?")
            params.append(repo_id)
        if commit_hash:
            conditions.append("commit_hash = ?")
            params.append(commit_hash)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_uncovered_functions(
        self,
        coverage_run_id: str,
        include_partial: bool = True,
    ) -> list[FunctionCoverage]:
        """Returns uncovered (and optionally partially covered) functions for a specific coverage run."""
        with self._get_connection() as conn:
            if include_partial:
                q = """
                    SELECT * FROM function_coverage
                    WHERE coverage_run_id = ? AND status IN ('UNCOVERED', 'PARTIALLY_COVERED')
                    ORDER BY status DESC, coverage_pct ASC
                """
            else:
                q = """
                    SELECT * FROM function_coverage
                    WHERE coverage_run_id = ? AND status = 'UNCOVERED'
                    ORDER BY qualified_name ASC
                """
            rows = conn.execute(q, (coverage_run_id,)).fetchall()
            results: list[FunctionCoverage] = []
            for r in rows:
                results.append(FunctionCoverage(
                    function_name=r["function_name"],
                    qualified_name=r["qualified_name"],
                    file_path=r["file_path"],
                    class_name=r["class_name"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    executable_lines=json.loads(r["executable_lines_json"] or "[]"),
                    executed_lines=json.loads(r["executed_lines_json"] or "[]"),
                    missing_lines=json.loads(r["missing_lines_json"] or "[]"),
                    coverage_pct=r["coverage_pct"],
                    status=FunctionStatus(r["status"]),
                    is_nested=bool(r["is_nested"]),
                    executing_tests=json.loads(r["executing_tests_json"] or "[]"),
                ))
            return results

    def get_test_mappings(
        self,
        coverage_run_id: str,
        test_id: Optional[str] = None,
    ) -> list[TestToCodeMapping]:
        """Returns test-to-code mapping entries for a coverage run."""
        with self._get_connection() as conn:
            if test_id:
                rows = conn.execute(
                    "SELECT * FROM test_code_mappings WHERE coverage_run_id = ? AND test_id = ?",
                    (coverage_run_id, test_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM test_code_mappings WHERE coverage_run_id = ?",
                    (coverage_run_id,),
                ).fetchall()

            mappings: list[TestToCodeMapping] = []
            for r in rows:
                mappings.append(TestToCodeMapping(
                    test_id=r["test_id"],
                    test_name=r["test_name"],
                    test_file=r["test_file"],
                    covered_files=json.loads(r["covered_files_json"] or "[]"),
                    covered_functions=json.loads(r["covered_functions_json"] or "[]"),
                    executed_lines_by_file=json.loads(r["executed_lines_json"] or "{}"),
                    evidence_type=CoverageEvidenceType(r["evidence_type"]),
                ))
            return mappings

    def delete_report(self, coverage_run_id: str) -> bool:
        """Deletes a coverage run and cascades to all child records and artifact files."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM test_code_mappings WHERE coverage_run_id = ?", (coverage_run_id,))
            conn.execute("DELETE FROM function_coverage WHERE coverage_run_id = ?", (coverage_run_id,))
            conn.execute("DELETE FROM file_coverage WHERE coverage_run_id = ?", (coverage_run_id,))
            conn.execute("DELETE FROM coverage_runs WHERE coverage_run_id = ?", (coverage_run_id,))
            conn.commit()

        json_path = self.reports_dir / f"{coverage_run_id}.json"
        if json_path.exists():
            try:
                json_path.unlink()
            except Exception:
                pass
        return True

