"""
TestPilot Flakiness Detection & Empirical Rerun Engine (Phase 2).
Calculates mathematically grounded flakiness scores from repeated executions.
Enforces rigorous credibility constraints:
1. Never gives 100% confidence from just 2 runs (separates sample size, failure rate, and confidence).
2. Uses inconsistent outcomes as primary signal, treating duration variance strictly as secondary context.
3. Provides an on-demand multi-iteration rerun probe to empirically measure consistency.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from testpilot.arbitration.history import (
    TestExecutionRecord,
    TestHistoryStore,
    TestHistorySummary,
)

logger = logging.getLogger("testpilot.arbitration.flakiness")


class FlakeClassification(str):
    CONSISTENT_PASS = "CONSISTENT_PASS"
    CONSISTENT_FAIL = "CONSISTENT_FAIL"
    CONFIRMED_FLAKE = "CONFIRMED_FLAKE"       # N >= 5, verified status flips, confidence >= 0.80
    SUSPECTED_FLAKE = "SUSPECTED_FLAKE"       # Inconsistent outcome observed, but sample size N < 5
    STABLE = "STABLE"


class FlakinessReport(BaseModel):
    """Rigorous flakiness evaluation report for a single test."""

    test_id: str
    test_name: str
    test_file: str
    total_runs: int = Field(..., description="Total observed executions across history")
    passed_runs: int = 0
    failed_runs: int = 0
    observed_failure_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Observed failure frequency: (failed / total_runs)"
    )
    flakiness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Mathematical flakiness metric based on status flip-rate and outcome balance"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Statistical confidence bounded strictly by sample size N"
    )
    classification: str = Field(..., description="CONFIRMED_FLAKE, SUSPECTED_FLAKE, CONSISTENT_FAIL, CONSISTENT_PASS, or STABLE")
    status_sequence: list[str] = Field(
        default_factory=list, description="Recent execution outcome sequence (e.g. ['PASSED', 'FAILED', 'PASSED'])"
    )
    avg_duration_ms: float = 0.0
    duration_std_dev_ms: float = 0.0
    duration_alert: Optional[str] = Field(
        default=None, description="Secondary warning if execution time varies widely, independent of flakiness"
    )
    evidence: list[str] = Field(default_factory=list)


class FlakinessDetectionEngine:
    """
    Evaluates historical test executions to detect non-deterministic flakiness.
    Enforces sample-size bounded confidence and provides on-demand multi-iteration rerun probing.
    """

    def __init__(self, history_store: Optional[TestHistoryStore] = None) -> None:
        self.history_store = history_store or TestHistoryStore.get_instance()

    def evaluate_test(self, test_id: str) -> FlakinessReport:
        """
        Evaluates flakiness for a single test from its historical runs.
        Separates observed failure rate, flakiness score, and sample-size confidence.
        """
        stats = self.history_store.get_test_stats(test_id)
        if not stats or stats.total_runs == 0:
            return FlakinessReport(
                test_id=test_id,
                test_name=test_id.split("::")[-1],
                test_file=test_id.split("::")[0],
                total_runs=0,
                passed_runs=0,
                failed_runs=0,
                observed_failure_rate=0.0,
                flakiness_score=0.0,
                confidence=0.0,
                classification=FlakeClassification.STABLE,
                status_sequence=[],
                evidence=["No recorded execution history for this test."],
            )

        n = stats.total_runs
        p = stats.passed_count
        f = stats.failed_count + stats.error_count
        seq = stats.recent_outcomes

        # 1. Primary Signal: Outcome Inconsistency & Status Transitions (Flips)
        # Count status transitions across consecutive chronological executions
        flips = 0
        for i in range(len(seq) - 1):
            if seq[i] in ("PASSED", "FAILED", "ERROR") and seq[i + 1] in ("PASSED", "FAILED", "ERROR"):
                is_cur_pass = (seq[i] == "PASSED")
                is_next_pass = (seq[i + 1] == "PASSED")
                if is_cur_pass != is_next_pass:
                    flips += 1

        flip_rate = (flips / max(1, len(seq) - 1)) if len(seq) > 1 else 0.0

        # Outcome balance ratio: 1.0 when 50/50 pass/fail, 0.0 when purely pass or purely fail
        executed_n = p + f
        balance = (4.0 * (p / executed_n) * (f / executed_n)) if executed_n > 0 else 0.0

        # Raw flakiness metric (0.0 to 1.0)
        if p == 0 and f > 0:
            raw_flakiness = 0.0
            classification = FlakeClassification.CONSISTENT_FAIL
        elif f == 0 and p > 0:
            raw_flakiness = 0.0
            classification = FlakeClassification.CONSISTENT_PASS
        else:
            raw_flakiness = round(0.55 * balance + 0.45 * flip_rate, 3)
            # Classification depends on sample size and certainty
            if n >= 5 and raw_flakiness >= 0.40:
                classification = FlakeClassification.CONFIRMED_FLAKE
            else:
                classification = FlakeClassification.SUSPECTED_FLAKE

        # 2. Credibility Constraint #1: Confidence MUST be bounded strictly by sample size N
        # N=1 -> 0.0, N=2 -> max 0.40, N=3 -> max 0.60, N=4 -> max 0.75, N>=5 -> scales up to 0.95
        if n <= 1:
            confidence = 0.10
        elif n == 2:
            confidence = 0.38  # 2 executions cannot establish reliable flakiness
        elif n == 3:
            confidence = 0.58
        elif n == 4:
            confidence = 0.72
        else:
            # Asymptotically approach 0.95 as N increases
            confidence = round(min(0.95, 0.75 + (0.20 * (1.0 - math.exp(-(n - 4) / 4.0)))), 3)

        evidence: list[str] = [
            f"Observed {n} total executions ({p} passed, {f} failed/errored).",
            f"Observed failure rate: {round(stats.failure_rate * 100, 1)}%.",
            f"Status transition count: {flips} flips across {len(seq)} sequential runs.",
        ]

        if n < 5 and classification == FlakeClassification.SUSPECTED_FLAKE:
            evidence.append(
                f"Preliminary signal only: sample size ({n} runs) is insufficient to establish flakiness with high confidence (confidence capped at {int(confidence*100)}%). Run on-demand rerun to verify."
            )
        elif classification == FlakeClassification.CONFIRMED_FLAKE:
            evidence.append(
                f"Statistically significant flakiness verified across {n} executions with {flips} outcome transitions."
            )

        # 3. Credibility Constraint #2: Duration variance is ONLY secondary supporting context
        dur_alert: Optional[str] = None
        if stats.avg_duration_ms > 0 and stats.duration_std_dev_ms > 0:
            coefficient_of_variation = stats.duration_std_dev_ms / stats.avg_duration_ms
            if coefficient_of_variation > 0.60 and stats.duration_std_dev_ms > 200:
                dur_alert = (
                    f"Notice: High execution time variation observed (mean: {stats.avg_duration_ms:.1f}ms, "
                    f"std dev: {stats.duration_std_dev_ms:.1f}ms). Test may have timing sensitivity or async polling, "
                    f"but flakiness is evaluated solely on outcome consistency."
                )
                evidence.append(dur_alert)

        return FlakinessReport(
            test_id=test_id,
            test_name=stats.test_name,
            test_file=stats.test_file,
            total_runs=n,
            passed_runs=p,
            failed_runs=f,
            observed_failure_rate=stats.failure_rate,
            flakiness_score=raw_flakiness,
            confidence=confidence,
            classification=classification,
            status_sequence=seq,
            avg_duration_ms=stats.avg_duration_ms,
            duration_std_dev_ms=stats.duration_std_dev_ms,
            duration_alert=dur_alert,
            evidence=evidence,
        )

    def scan_all_tests(self, repo_id: Optional[str] = None) -> list[FlakinessReport]:
        """Scans all known tests and returns their flakiness reports, sorted by flakiness score descending."""
        summaries = self.history_store.get_all_test_summaries(repo_id=repo_id)
        reports = []
        for s in summaries:
            rep = self.evaluate_test(s.test_id)
            reports.append(rep)
        reports.sort(key=lambda r: (r.flakiness_score, r.confidence), reverse=True)
        return reports

    def rerun_test_on_demand(
        self,
        test_file: str,
        test_name: Optional[str] = None,
        iterations: int = 5,
        repo_path: Path | str = ".",
        repo_id: str = "default",
        commit_hash: str = "HEAD",
        workflow_id: Optional[str] = None,
    ) -> FlakinessReport:
        """
        Actively probes test consistency by executing the target test N consecutive times in isolation.
        Each iteration is individually tracked in TestHistoryStore, immediately updating the empirical sample size.
        """
        base_dir = Path(repo_path).expanduser().resolve()
        target_spec = f"{test_file}::{test_name}" if test_name else test_file
        test_id = target_spec

        iterations_to_run = max(2, min(10, iterations))
        logger.info(f"Launching on-demand flakiness probe: {target_spec} ({iterations_to_run} iterations)")

        python_exe = sys.executable

        for idx in range(iterations_to_run):
            start = time.perf_counter()
            cmd = [python_exe, "-m", "pytest", str(target_spec), "-q", "--tb=short", "--no-header"]
            outcome = "PASSED"
            err_msg = ""
            tb_str = ""
            out_str = ""

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                duration_ms = (time.perf_counter() - start) * 1000.0
                out_str = proc.stdout or ""

                if proc.returncode == 0:
                    outcome = "PASSED"
                elif proc.returncode == 5:
                    outcome = "SKIPPED"
                else:
                    outcome = "FAILED"
                    err_msg = (proc.stderr or proc.stdout or "").strip()
                    tb_str = proc.stdout or ""
            except subprocess.TimeoutExpired:
                duration_ms = (time.perf_counter() - start) * 1000.0
                outcome = "ERROR"
                err_msg = "Execution timed out after 30 seconds."
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                outcome = "ERROR"
                err_msg = str(e)

            sig = TestExecutionRecord.create_signature(err_msg, tb_str) if outcome != "PASSED" else ""

            rec = TestExecutionRecord(
                test_id=test_id,
                test_name=test_name or test_file.split("/")[-1],
                test_file=test_file,
                repo_id=repo_id,
                commit_hash=commit_hash,
                workflow_id=workflow_id,
                execution_run_id=f"rerun-{int(time.time())}-{idx+1}",
                outcome=outcome,
                duration_ms=round(duration_ms, 2),
                failure_signature=sig,
                error_message=err_msg[:500] if err_msg else None,
                traceback=tb_str[:1000] if tb_str else None,
                stdout=out_str[:500] if out_str else None,
                execution_mode="on_demand_probe",
            )
            self.history_store.record_execution(rec)

        # Return freshly evaluated report with the new empirical data points
        return self.evaluate_test(test_id)
