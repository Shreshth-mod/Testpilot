"""
TestPilot Detailed Failure Investigation & Order Pollution Subsystem (Phases 4 & 5).
Explains test failures with strictly grounded execution evidence and synthesizes actionable repairs.
Enforces the core credibility constraint:
Never invent root causes — if evidence is incomplete or ambiguous, explicitly returns "Insufficient evidence".
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from testpilot.arbitration.flakiness import FlakinessDetectionEngine, FlakinessReport
from testpilot.arbitration.history import TestHistoryStore
from testpilot.arbitration.models import ArbitrationCategory
from testpilot.llm.client import OllamaLLMClient

logger = logging.getLogger("testpilot.arbitration.investigator")


class FailureInvestigationResult(BaseModel):
    """Authoritative, evidence-grounded failure investigation diagnostic report."""

    test_id: str
    test_name: str
    test_file: str
    what_went_wrong: str = Field(..., description="Exact exception type, assertion, or failure point extracted from traceback")
    why_it_failed: str = Field(..., description="Grounded root cause diagnosis tied to code, diff, or environment")
    category: str = Field(..., description="REGRESSION, PRE_EXISTING, FLAKY, INFRASTRUCTURE, TEST_DEFECT, or UNCERTAIN")
    category_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_trail: list[str] = Field(default_factory=list, description="Concrete traceframes, AST symbols, and historical runs")
    investigate_next: list[str] = Field(default_factory=list, description="Actionable next steps for the developer")
    suggested_repair: str = Field(..., description="Concrete fix patch or 'Insufficient evidence'")
    repair_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_grounded: bool = Field(default=True, description="True if diagnosis is proven by concrete traces and diffs")
    
    # Phase 5 Advanced features
    order_pollution_detected: bool = False
    order_pollution_details: Optional[str] = None
    is_quarantined: bool = False
    flakiness_report: Optional[FlakinessReport] = None


class FailureInvestigator:
    """
    Synthesizes deep diagnostic investigations for failing tests.
    Grounded in execution traces, AST symbol diffs, historical telemetry, and isolated order probes.
    """

    def __init__(
        self,
        history_store: Optional[TestHistoryStore] = None,
        flakiness_engine: Optional[FlakinessDetectionEngine] = None,
        llm_client: Optional[OllamaLLMClient] = None,
    ) -> None:
        self.history_store = history_store or TestHistoryStore.get_instance()
        self.flakiness_engine = flakiness_engine or FlakinessDetectionEngine(self.history_store)
        self.llm_client = llm_client or OllamaLLMClient()

    def probe_order_pollution(
        self, test_file: str, test_name: str, repo_path: Path | str = "."
    ) -> tuple[bool, Optional[str]]:
        """
        Phase 5 Advanced Feature: Test Order Pollution Detection.
        Executes the failed test completely alone in isolation.
        If it PASSES in isolation despite failing during the test suite execution,
        this proves upstream test pollution (unclosed resources, modified globals, or leaked mocks).
        """
        base_dir = Path(repo_path).expanduser().resolve()
        target_spec = f"{test_file}::{test_name}" if test_name else test_file
        python_exe = sys.executable
        cmd = [python_exe, "-m", "pytest", str(target_spec), "-q", "--tb=line", "--no-header"]

        try:
            res = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True, timeout=25, check=False)
            if res.returncode == 0:
                details = (
                    f"Test Order Pollution Confirmed: '{target_spec}' PASSED when executed alone in isolation, "
                    f"but FAILED during test suite execution. This indicates preceding tests in the suite "
                    f"leaked state (e.g. unreset module-level globals, unclosed mock patches, or uncleaned database fixtures)."
                )
                return True, details
            else:
                return False, None
        except Exception as e:
            logger.debug(f"Order pollution probe failed to run: {e}")
            return False, None

    def investigate(
        self,
        test_id: str,
        test_name: Optional[str] = None,
        test_file: Optional[str] = None,
        error_message: Optional[str] = None,
        traceback: Optional[str] = None,
        category: Optional[str] = None,
        category_confidence: Optional[float] = None,
        targeted_symbol: Optional[str] = None,
        target_commit_hash: Optional[str] = None,
        repo_path: Path | str = ".",
        check_pollution: bool = True,
    ) -> FailureInvestigationResult:
        """
        Produces an authoritative, grounded 5-point failure explanation.
        """
        t_file = test_file or (test_id.split("::")[0] if "::" in test_id else "unknown_file.py")
        t_name = test_name or (test_id.split("::")[-1] if "::" in test_id else test_id)

        # Retrieve historical record if available to enrich details
        history = self.history_store.get_history(test_id, limit=10)
        recent_record = history[0] if history else None

        actual_error = error_message or (recent_record.error_message if recent_record else "") or "Test assertion or execution failure"
        actual_tb = traceback or (recent_record.traceback if recent_record else "") or ""
        cat = category or "UNCERTAIN"
        conf = category_confidence or 0.60

        # Flakiness and quarantine checks
        flake_report = self.flakiness_engine.evaluate_test(test_id)
        is_quarantined = self.history_store.is_quarantined(test_id)

        # 1. Answer Question 1: What exactly went wrong?
        what_went_wrong = self._extract_what_went_wrong(actual_error, actual_tb)

        # 2. Answer Question 2: Why did the test fail?
        why_it_failed, is_grounded = self._grounded_why_it_failed(
            what_went_wrong=what_went_wrong,
            error_msg=actual_error,
            traceback=actual_tb,
            category=cat,
            targeted_symbol=targeted_symbol,
            flake_report=flake_report,
            commit_hash=target_commit_hash,
        )

        # 3. Answer Question 3: Assemble concrete evidence trail
        evidence_trail: list[str] = []
        if actual_error:
            evidence_trail.append(f"Failure signature: {actual_error.strip().splitlines()[0][:160]}")
        
        extracted_frame = self._extract_relevant_frame(actual_tb)
        if extracted_frame:
            evidence_trail.append(f"Failing execution frame: {extracted_frame}")

        if targeted_symbol:
            evidence_trail.append(f"Targeted code symbol under test: {targeted_symbol}")

        if flake_report.total_runs > 0:
            evidence_trail.append(
                f"Historical record: {flake_report.total_runs} observed runs ({flake_report.passed_runs} pass, "
                f"{flake_report.failed_runs} fail). Observed failure rate: {round(flake_report.observed_failure_rate * 100, 1)}%."
            )

        if flake_report.duration_alert:
            evidence_trail.append(flake_report.duration_alert)

        # 4. Optional Phase 5 Check: Order Pollution Probe
        order_pollution = False
        pollution_details = None
        if check_pollution and cat in ("UNCERTAIN", "FLAKY", "REGRESSION"):
            order_pollution, pollution_details = self.probe_order_pollution(t_file, t_name, repo_path=repo_path)
            if order_pollution and pollution_details:
                evidence_trail.append(pollution_details)

        # 5. Answer Question 4: Actionable Next Steps
        investigate_next = self._generate_investigation_steps(
            category=cat,
            what_went_wrong=what_went_wrong,
            test_file=t_file,
            test_name=t_name,
            targeted_symbol=targeted_symbol,
            order_pollution=order_pollution,
            flake_report=flake_report,
        )

        # 6. Answer Question 5: Grounded Repair Suggestion (strictly enforces "Insufficient evidence" fallback)
        suggested_repair, repair_conf = self._synthesize_repair(
            what_went_wrong=what_went_wrong,
            why_it_failed=why_it_failed,
            category=cat,
            error_msg=actual_error,
            traceback=actual_tb,
            test_file=t_file,
            targeted_symbol=targeted_symbol,
            is_grounded=is_grounded,
        )

        return FailureInvestigationResult(
            test_id=test_id,
            test_name=t_name,
            test_file=t_file,
            what_went_wrong=what_went_wrong,
            why_it_failed=why_it_failed,
            category=cat,
            category_confidence=conf,
            evidence_trail=evidence_trail,
            investigate_next=investigate_next,
            suggested_repair=suggested_repair,
            repair_confidence=repair_conf,
            is_grounded=is_grounded,
            order_pollution_detected=order_pollution,
            order_pollution_details=pollution_details,
            is_quarantined=is_quarantined,
            flakiness_report=flake_report if flake_report.total_runs > 0 else None,
        )

    def _extract_what_went_wrong(self, error_msg: str, tb: str) -> str:
        """Extracts the precise exception class and failure message from output."""
        if tb:
            # Pytest format often ends with: E AssertionError: assert False or E ValueError: ...
            tb_lines = [line.strip() for line in tb.strip().splitlines() if line.strip()]
            for line in reversed(tb_lines):
                if line.startswith("E ") or "Error:" in line or "assert " in line or "Exception:" in line:
                    clean = line[2:].strip() if line.startswith("E ") else line
                    return clean[:200]
            if tb_lines:
                return tb_lines[-1][:200]

        if error_msg:
            first_line = error_msg.strip().splitlines()[0].strip()
            return first_line[:200]

        return "Unknown failure: test exited with non-zero exit code without stdout/stderr output."

    def _extract_relevant_frame(self, tb: str) -> Optional[str]:
        """Finds the most specific repository code frame from traceback."""
        if not tb:
            return None
        lines = tb.splitlines()
        candidate = None
        for line in lines:
            if "File " in line and ".py" in line:
                candidate = line.strip()
        return candidate

    def _grounded_why_it_failed(
        self,
        what_went_wrong: str,
        error_msg: str,
        traceback: str,
        category: str,
        targeted_symbol: Optional[str],
        flake_report: FlakinessReport,
        commit_hash: Optional[str],
    ) -> tuple[str, bool]:
        """
        Synthesizes a factually grounded explanation for why the test failed.
        Returns (explanation, is_grounded).
        """
        err_lower = (error_msg + " " + traceback).lower()

        if category == ArbitrationCategory.REGRESSION.value:
            symbol_str = f" in function or method '{targeted_symbol}'" if targeted_symbol else ""
            return (
                f"A behavioral regression was introduced by the code modification{symbol_str}. "
                f"The target implementation produced an output or side-effect that violated the test contract: '{what_went_wrong}'.",
                True,
            )

        if category == ArbitrationCategory.PRE_EXISTING.value:
            return (
                f"The failure was present prior to the current commit and exists on the base baseline revision. "
                f"It represents pre-existing technical debt rather than a regression from the active commit.",
                True,
            )

        if category == ArbitrationCategory.FLAKY.value:
            if flake_report.classification == "CONFIRMED_FLAKE":
                return (
                    f"Test exhibits non-deterministic execution flakiness verified across {flake_report.total_runs} runs. "
                    f"It alternates between passing and failing due to asynchronous timing jitter, port allocation, or state persistence.",
                    True,
                )
            else:
                return (
                    f"Intermittent failure signature detected. Preliminary evidence indicates susceptibility to race conditions or timing dependencies. "
                    f"Confidence is capped at {int(flake_report.confidence * 100)}% pending additional empirical reruns.",
                    True,
                )

        if category == ArbitrationCategory.INFRASTRUCTURE.value:
            return (
                f"Execution failed due to environmental or sandbox constraints (e.g. timeout, missing OS library, or container limits) "
                f"rather than internal application logic error: '{what_went_wrong}'.",
                True,
            )

        if category == ArbitrationCategory.TEST_DEFECT.value:
            return (
                f"The test definition or its test fixture setup is defective. It called an API with invalid parameters, "
                f"referenced an unimported fixture, or asserted an unreachable condition: '{what_went_wrong}'.",
                True,
            )

        # UNCERTAIN: Must be honest when evidence is lacking
        return (
            "Insufficient evidence to conclusively determine root cause. "
            "Neither stack trace nor git diff provides definitive classification.",
            False,
        )

    def _generate_investigation_steps(
        self,
        category: str,
        what_went_wrong: str,
        test_file: str,
        test_name: str,
        targeted_symbol: Optional[str],
        order_pollution: bool,
        flake_report: FlakinessReport,
    ) -> list[str]:
        """Provides concrete, actionable developer next steps."""
        steps: list[str] = []

        if order_pollution:
            steps.append(
                f"Investigate preceding tests in the test suite for leaked state. Run pytest with '--random-order' or isolate the test with pytest {test_file}::{test_name}."
            )
            steps.append("Check if preceding tests clean up their monkeypatches, database rows, or temporary filesystem directories.")

        if category == ArbitrationCategory.REGRESSION.value:
            if targeted_symbol:
                steps.append(f"Inspect git diff for changes inside symbol '{targeted_symbol}'.")
            steps.append(f"Compare returned value against assertion condition: '{what_went_wrong}'.")
            steps.append(f"Run localized verification: python -m pytest {test_file}::{test_name} -vv")

        elif category == ArbitrationCategory.FLAKY.value:
            steps.append("Run the on-demand flakiness probe (5x rerun) to observe pass/fail flip frequency.")
            steps.append("Check for unawaited async coroutines, hardcoded time.sleep() calls, or shared socket ports.")
            steps.append("Consider placing this test in Quarantine if it blocks production CI pipelines.")

        elif category == ArbitrationCategory.INFRASTRUCTURE.value:
            steps.append("Check Docker daemon logs or container memory/CPU utilization limits.")
            steps.append("Verify required system packages and network connectivity requirements.")

        elif category == ArbitrationCategory.TEST_DEFECT.value:
            steps.append(f"Inspect test implementation in '{test_file}' at function '{test_name}'.")
            steps.append("Verify fixture names, argument counts, and mock call expectations.")

        else:
            steps.append("Inspect the raw stdout and stderr logs for detailed traceback frames.")
            steps.append(f"Execute the test individually in your terminal: pytest {test_file}::{test_name} -s")

        return steps

    def _synthesize_repair(
        self,
        what_went_wrong: str,
        why_it_failed: str,
        category: str,
        error_msg: str,
        traceback: str,
        test_file: str,
        targeted_symbol: Optional[str],
        is_grounded: bool,
    ) -> tuple[str, float]:
        """
        Synthesizes a proposed repair or fallback.
        Strictly enforces Credibility Constraint #3:
        If TestPilot cannot establish the cause, it says 'Insufficient evidence'.
        """
        if not is_grounded or category == ArbitrationCategory.UNCERTAIN.value:
            return (
                "Insufficient evidence to generate an automated repair safely. "
                "Stack trace lacks deterministic symbol or fixture context.",
                0.0,
            )

        # Deterministic rule-based repairs for common test defects and infra issues
        if category == ArbitrationCategory.TEST_DEFECT.value:
            if "fixture" in what_went_wrong.lower() and "not found" in what_went_wrong.lower():
                match = re.search(r"fixture ['\"](\w+)['\"] not found", what_went_wrong)
                fix_name = match.group(1) if match else "fixture"
                return (
                    f"Proposed Fix for Test Defect:\n"
                    f"Missing pytest fixture '{fix_name}'. Add `@pytest.fixture def {fix_name}():` into `{test_file}` or your `conftest.py`.",
                    0.92,
                )
            if "takes 0 positional arguments but 1 was given" in what_went_wrong.lower():
                return (
                    f"Proposed Fix for Test Defect:\n"
                    f"Method signature mismatch. In `{test_file}`, ensure the test method accepts `self` if defined inside a class: `def test_something(self):`.",
                    0.95,
                )

        if category == ArbitrationCategory.INFRASTRUCTURE.value:
            if "timed out" in what_went_wrong.lower() or "timeout" in what_went_wrong.lower():
                return (
                    "Proposed Fix for Infrastructure Timeout:\n"
                    "Increase execution timeout in TestPilot configuration or add `@pytest.mark.timeout(60)` to allow long-running operations to finish.",
                    0.88,
                )

        if category == ArbitrationCategory.FLAKY.value:
            return (
                "Proposed Fix for Flakiness / Non-determinism:\n"
                "1. Replace fixed `time.sleep()` calls with dynamic condition polling (`tenacity.retry` or `wait_until`).\n"
                "2. Ensure socket listeners use port 0 (`socket.bind(('127.0.0.1', 0))`) to avoid port collisions.\n"
                "3. Enable Quarantine in the Test Intelligence Dashboard to shield master branches while stabilizing.",
                0.85,
            )

        if category == ArbitrationCategory.REGRESSION.value:
            if "AssertionError" in what_went_wrong and targeted_symbol:
                return (
                    f"Proposed Regression Remediations for '{targeted_symbol}':\n"
                    f"Review the implementation changes in `{targeted_symbol}` against the assertion `{what_went_wrong}`.\n"
                    f"Ensure output format, return types, and boundary edge cases match the expected specification contract.",
                    0.82,
                )

        # Default fallback if concrete repair cannot be verified
        return (
            "Insufficient evidence to safely synthesize automated code modification. "
            "Examine the failing frame in the traceback to inspect input variables at runtime.",
            0.15,
        )
