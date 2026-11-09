"""
TestPilot Historical Test Execution Tracking Subsystem (Phase 1).
Maintains an indexed SQLite datastore recording execution outcomes,
commit metadata, execution durations, and failure signatures over time.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("testpilot.arbitration.history")


class TestExecutionRecord(BaseModel):
    """Immutable audit record for a single test execution event."""

    id: Optional[int] = None
    test_id: str = Field(..., description="Canonical qualified test identifier (file::function)")
    test_name: str = Field(..., description="Test function or method name")
    test_file: str = Field(..., description="Repository-relative test file path")
    repo_id: str = Field(default="default", description="Repository ID or sanitized name")
    commit_hash: str = Field(default="HEAD", description="Git commit hash during execution")
    workflow_id: Optional[str] = None
    execution_run_id: Optional[str] = None
    outcome: str = Field(..., description="PASSED, FAILED, ERROR, or SKIPPED")
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    failure_signature: str = Field(default="", description="Normalized failure classification signature")
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    stdout: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of execution",
    )
    execution_mode: str = Field(default="local_subprocess", description="local_subprocess, docker, rerun, or dry_run")

    @classmethod
    def create_signature(cls, error_msg: Optional[str], tb: Optional[str]) -> str:
        """Derives a normalized signature string from error message and top traceback frame."""
        if not error_msg and not tb:
            return ""
        raw = (error_msg or "").strip().splitlines()[0] if error_msg else ""
        if tb:
            # Extract last line of traceback (exception class + message)
            tb_lines = [line.strip() for line in tb.strip().splitlines() if line.strip()]
            if tb_lines:
                raw = tb_lines[-1]
        # Normalize whitespace and truncate
        clean = " ".join(raw.split())[:200]
        return clean


class TestHistorySummary(BaseModel):
    """Aggregated empirical statistical summary of a test's execution history."""

    test_id: str
    test_name: str
    test_file: str
    repo_id: str
    total_runs: int = 0
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of non-skipped runs that failed/errored")
    avg_duration_ms: float = 0.0
    duration_std_dev_ms: float = 0.0
    recent_outcomes: list[str] = Field(default_factory=list, description="Chronological sequence of recent outcomes")
    distinct_failure_signatures: list[str] = Field(default_factory=list)
    unique_commits: list[str] = Field(default_factory=list)
    last_run_timestamp: Optional[str] = None
    last_run_outcome: Optional[str] = None


class TestHistoryStore:
    """
    Thread-safe, indexed SQLite persistence store for historical test execution runs.
    Stores and queries execution telemetry for flakiness, regression, and reliability analysis.
    """

    _instance: Optional[TestHistoryStore] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path(".testpilot/test_history.db").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path | str] = None) -> TestHistoryStore:
        """Singleton accessor for global TestHistoryStore."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables and high-cardinality query indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    workflow_id TEXT,
                    execution_run_id TEXT,
                    outcome TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    failure_signature TEXT,
                    error_message TEXT,
                    traceback TEXT,
                    stdout TEXT,
                    timestamp TEXT NOT NULL,
                    execution_mode TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_test_id ON test_execution_history(test_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_repo_id ON test_execution_history(repo_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_commit ON test_execution_history(commit_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_timestamp ON test_execution_history(timestamp)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_quarantine (
                    test_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    author TEXT NOT NULL,
                    quarantined_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.commit()

    def record_execution(self, record: TestExecutionRecord) -> int:
        """Inserts a single test execution record."""
        return self.record_batch([record])

    def record_batch(self, records: list[TestExecutionRecord]) -> int:
        """Batch inserts test execution records atomically inside a single transaction."""
        if not records:
            return 0

        rows = [
            (
                r.test_id,
                r.test_name,
                r.test_file,
                r.repo_id,
                r.commit_hash,
                r.workflow_id,
                r.execution_run_id,
                r.outcome.upper(),
                float(r.duration_ms),
                r.failure_signature or "",
                r.error_message or "",
                r.traceback or "",
                r.stdout or "",
                r.timestamp,
                r.execution_mode,
            )
            for r in records
        ]

        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO test_execution_history (
                    test_id, test_name, test_file, repo_id, commit_hash,
                    workflow_id, execution_run_id, outcome, duration_ms,
                    failure_signature, error_message, traceback, stdout,
                    timestamp, execution_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return cursor.rowcount

    def get_history(self, test_id: str, limit: int = 50) -> list[TestExecutionRecord]:
        """Returns chronological execution events for a specific test_id, most recent first."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM test_execution_history
                WHERE test_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (test_id, limit),
            )
            results = []
            for row in cursor.fetchall():
                results.append(
                    TestExecutionRecord(
                        id=row["id"],
                        test_id=row["test_id"],
                        test_name=row["test_name"],
                        test_file=row["test_file"],
                        repo_id=row["repo_id"],
                        commit_hash=row["commit_hash"],
                        workflow_id=row["workflow_id"],
                        execution_run_id=row["execution_run_id"],
                        outcome=row["outcome"],
                        duration_ms=row["duration_ms"],
                        failure_signature=row["failure_signature"],
                        error_message=row["error_message"],
                        traceback=row["traceback"],
                        stdout=row["stdout"],
                        timestamp=row["timestamp"],
                        execution_mode=row["execution_mode"],
                    )
                )
            return results

    def get_test_stats(self, test_id: str) -> Optional[TestHistorySummary]:
        """Computes statistical metrics (failure rate, duration variance, status sequence) for a test."""
        history = self.get_history(test_id, limit=100)
        if not history:
            return None

        # Chronological order (oldest to newest) for sequence analysis
        chronological = list(reversed(history))

        total_runs = len(chronological)
        passed_count = sum(1 for r in chronological if r.outcome == "PASSED")
        failed_count = sum(1 for r in chronological if r.outcome == "FAILED")
        error_count = sum(1 for r in chronological if r.outcome == "ERROR")
        skipped_count = sum(1 for r in chronological if r.outcome == "SKIPPED")

        executed_count = passed_count + failed_count + error_count
        failure_rate = round((failed_count + error_count) / max(1, executed_count), 4)

        durations = [r.duration_ms for r in chronological if r.duration_ms > 0]
        avg_dur = round(sum(durations) / max(1, len(durations)), 2) if durations else 0.0

        variance = (
            sum((d - avg_dur) ** 2 for d in durations) / len(durations)
            if len(durations) > 1
            else 0.0
        )
        std_dev = round(math.sqrt(variance), 2)

        recent_outcomes = [r.outcome for r in chronological[-15:]]
        distinct_signatures = list(
            {r.failure_signature for r in chronological if r.failure_signature}
        )
        unique_commits = list({r.commit_hash for r in chronological if r.commit_hash})

        last_rec = chronological[-1]

        return TestHistorySummary(
            test_id=test_id,
            test_name=last_rec.test_name,
            test_file=last_rec.test_file,
            repo_id=last_rec.repo_id,
            total_runs=total_runs,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            skipped_count=skipped_count,
            failure_rate=failure_rate,
            avg_duration_ms=avg_dur,
            duration_std_dev_ms=std_dev,
            recent_outcomes=recent_outcomes,
            distinct_failure_signatures=distinct_signatures,
            unique_commits=unique_commits,
            last_run_timestamp=last_rec.timestamp,
            last_run_outcome=last_rec.outcome,
        )

    def get_all_test_summaries(
        self, repo_id: Optional[str] = None, limit: int = 200
    ) -> list[TestHistorySummary]:
        """Returns statistical summaries for all unique test_ids recorded."""
        with self._get_connection() as conn:
            if repo_id:
                cursor = conn.execute(
                    "SELECT DISTINCT test_id FROM test_execution_history WHERE repo_id = ? LIMIT ?",
                    (repo_id, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT DISTINCT test_id FROM test_execution_history LIMIT ?",
                    (limit,),
                )
            test_ids = [row["test_id"] for row in cursor.fetchall()]

        summaries = []
        for tid in test_ids:
            stat = self.get_test_stats(tid)
            if stat:
                summaries.append(stat)
        return summaries

    def get_recent_runs(
        self, repo_id: Optional[str] = None, limit: int = 100
    ) -> list[TestExecutionRecord]:
        """Returns the most recent global test execution runs."""
        with self._get_connection() as conn:
            if repo_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM test_execution_history
                    WHERE repo_id = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (repo_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM test_execution_history
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            return [
                TestExecutionRecord(
                    id=row["id"],
                    test_id=row["test_id"],
                    test_name=row["test_name"],
                    test_file=row["test_file"],
                    repo_id=row["repo_id"],
                    commit_hash=row["commit_hash"],
                    workflow_id=row["workflow_id"],
                    execution_run_id=row["execution_run_id"],
                    outcome=row["outcome"],
                    duration_ms=row["duration_ms"],
                    failure_signature=row["failure_signature"],
                    error_message=row["error_message"],
                    traceback=row["traceback"],
                    stdout=row["stdout"],
                    timestamp=row["timestamp"],
                    execution_mode=row["execution_mode"],
                )
                for row in cursor.fetchall()
            ]

    def clear_history(self, repo_id: Optional[str] = None) -> int:
        """Clears execution history records."""
        with self._get_connection() as conn:
            if repo_id:
                cur = conn.execute("DELETE FROM test_execution_history WHERE repo_id = ?", (repo_id,))
            else:
                cur = conn.execute("DELETE FROM test_execution_history")
            conn.commit()
            return cur.rowcount

    def quarantine_test(self, test_id: str, reason: str = "Suspected flaky or non-deterministic", author: str = "TestPilot Intelligence") -> bool:
        """Adds or updates a test in quarantine status."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO test_quarantine (test_id, reason, author, quarantined_at, is_active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(test_id) DO UPDATE SET
                    reason = excluded.reason,
                    author = excluded.author,
                    quarantined_at = excluded.quarantined_at,
                    is_active = 1
                """,
                (test_id, reason, author, now),
            )
            conn.commit()
            return True

    def unquarantine_test(self, test_id: str) -> bool:
        """Removes a test from active quarantine."""
        with self._get_connection() as conn:
            conn.execute("UPDATE test_quarantine SET is_active = 0 WHERE test_id = ?", (test_id,))
            conn.commit()
            return True

    def is_quarantined(self, test_id: str) -> bool:
        """Checks if a test is currently quarantined."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT is_active FROM test_quarantine WHERE test_id = ? AND is_active = 1",
                (test_id,),
            )
            return cur.fetchone() is not None

    def get_quarantined_tests(self) -> list[dict[str, Any]]:
        """Returns all currently quarantined tests."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT test_id, reason, author, quarantined_at FROM test_quarantine WHERE is_active = 1 ORDER BY quarantined_at DESC")
            return [
                {
                    "test_id": r["test_id"],
                    "reason": r["reason"],
                    "author": r["author"],
                    "quarantined_at": r["quarantined_at"],
                }
                for r in cur.fetchall()
            ]

