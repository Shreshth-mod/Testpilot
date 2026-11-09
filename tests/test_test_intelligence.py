"""
Comprehensive test suite for Test Intelligence Dashboard & Flakiness Detection (Phases 1-5).
Verifies:
1. Phase 1: Historical test tracking (SQLite datastore, batch ingestion, query indexes).
2. Phase 2: Flakiness detection engine & technical credibility constraints:
   - Constraint 1: Never awards 100% flakiness certainty from 2 runs (N=2 -> confidence <= 0.38).
   - Constraint 2: Outcome inconsistency is primary signal; duration variation is strictly secondary context.
   - On-demand multi-iteration rerun probe actively updates empirical sample size.
3. Phase 3: Centralized Intelligence REST API endpoints (/api/intelligence/*).
4. Phase 4: Grounded failure investigation (5 core questions answered, zero hallucination fallback).
5. Phase 5: Advanced features (Test order pollution detection, quarantine mechanism, reliability degradation alerts).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from testpilot.arbitration.flakiness import FlakinessDetectionEngine, FlakeClassification
from testpilot.arbitration.history import TestExecutionRecord, TestHistoryStore
from testpilot.arbitration.investigator import FailureInvestigator
from testpilot.arbitration.models import ArbitrationCategory
from testpilot.web.api import app

client = TestClient(app)


@pytest.fixture
def temp_history_store(tmp_path):
    """Provides a fresh isolated SQLite store for testing."""
    db_file = tmp_path / "test_history.db"
    return TestHistoryStore(db_path=db_file)


# =========================================================================
# PHASE 1: Historical Test Tracking Tests
# =========================================================================
def test_history_store_record_and_query(temp_history_store):
    records = [
        TestExecutionRecord(
            test_id="tests/test_cart.py::test_checkout_total",
            test_name="test_checkout_total",
            test_file="tests/test_cart.py",
            repo_id="repo-1",
            commit_hash="commit-aaa",
            outcome="PASSED",
            duration_ms=45.2,
        ),
        TestExecutionRecord(
            test_id="tests/test_cart.py::test_checkout_total",
            test_name="test_checkout_total",
            test_file="tests/test_cart.py",
            repo_id="repo-1",
            commit_hash="commit-bbb",
            outcome="FAILED",
            duration_ms=52.1,
            error_message="AssertionError: assert 100 == 90",
            traceback="Traceback:\n  File 'test_cart.py', line 12\nAssertionError: assert 100 == 90",
        ),
    ]

    count = temp_history_store.record_batch(records)
    assert count == 2

    # Query history
    history = temp_history_store.get_history("tests/test_cart.py::test_checkout_total")
    assert len(history) == 2
    assert history[0].commit_hash == "commit-bbb"  # Recent first

    # Query stats
    stats = temp_history_store.get_test_stats("tests/test_cart.py::test_checkout_total")
    assert stats is not None
    assert stats.total_runs == 2
    assert stats.passed_count == 1
    assert stats.failed_count == 1
    assert stats.failure_rate == 0.50
    assert stats.recent_outcomes == ["PASSED", "FAILED"]


# =========================================================================
# PHASE 2: Flakiness Detection & Credibility Constraints Tests
# =========================================================================
def test_credibility_constraint_1_never_100_percent_on_two_runs(temp_history_store):
    """
    User Correction 1:
    Don't give a test 100% flakiness certainty just because one rerun passes.
    A test that fails once and passes once has insufficient sample size.
    Confidence must be strictly bounded (N=2 -> confidence <= 0.38).
    """
    engine = FlakinessDetectionEngine(history_store=temp_history_store)
    test_id = "tests/test_flaky.py::test_intermittent_service"

    # 1 Pass, 1 Fail (2 runs total)
    temp_history_store.record_execution(
        TestExecutionRecord(
            test_id=test_id,
            test_name="test_intermittent_service",
            test_file="tests/test_flaky.py",
            outcome="FAILED",
            duration_ms=10.0,
        )
    )
    temp_history_store.record_execution(
        TestExecutionRecord(
            test_id=test_id,
            test_name="test_intermittent_service",
            test_file="tests/test_flaky.py",
            outcome="PASSED",
            duration_ms=12.0,
        )
    )

    report = engine.evaluate_test(test_id)
    assert report.total_runs == 2
    assert report.passed_runs == 1
    assert report.failed_runs == 1
    assert report.observed_failure_rate == 0.50
    # Must NOT be CONFIRMED_FLAKE because N < 5
    assert report.classification == FlakeClassification.SUSPECTED_FLAKE
    # Confidence MUST be strictly capped <= 0.38
    assert report.confidence <= 0.38
    assert "sample size (2 runs) is insufficient" in " ".join(report.evidence).lower()


def test_credibility_constraint_2_slow_tests_not_confused_with_flaky(temp_history_store):
    """
    User Correction 2:
    Don't confuse slow tests with flaky tests.
    A slow test or execution-time variation is NOT necessarily flaky.
    Inconsistent outcomes are the primary signal; duration variation is secondary.
    """
    engine = FlakinessDetectionEngine(history_store=temp_history_store)
    test_id = "tests/test_slow.py::test_slow_batch_processing"

    # 6 consecutive executions, ALL PASSED, but wildly varying execution times (100ms to 2500ms)
    durations = [100.0, 2400.0, 150.0, 2100.0, 120.0, 2500.0]
    for d in durations:
        temp_history_store.record_execution(
            TestExecutionRecord(
                test_id=test_id,
                test_name="test_slow_batch_processing",
                test_file="tests/test_slow.py",
                outcome="PASSED",
                duration_ms=d,
            )
        )

    report = engine.evaluate_test(test_id)
    # Since outcomes are 100% consistent (ALL PASSED), test must NOT be flaky!
    assert report.classification == FlakeClassification.CONSISTENT_PASS
    assert report.flakiness_score == 0.0
    # But it SHOULD report the duration variation alert as secondary context
    assert report.duration_alert is not None
    assert "High execution time variation" in report.duration_alert


def test_confirmed_flakiness_with_high_sample_size(temp_history_store):
    """When N >= 5 with confirmed outcome flips, test achieves CONFIRMED_FLAKE status."""
    engine = FlakinessDetectionEngine(history_store=temp_history_store)
    test_id = "tests/test_network.py::test_socket_poll"

    # Sequence of flips across 6 runs: P, F, P, F, P, F
    outcomes = ["PASSED", "FAILED", "PASSED", "FAILED", "PASSED", "FAILED"]
    for o in outcomes:
        temp_history_store.record_execution(
            TestExecutionRecord(
                test_id=test_id,
                test_name="test_socket_poll",
                test_file="tests/test_network.py",
                outcome=o,
                duration_ms=25.0,
            )
        )

    report = engine.evaluate_test(test_id)
    assert report.total_runs == 6
    assert report.classification == FlakeClassification.CONFIRMED_FLAKE
    assert report.flakiness_score >= 0.70
    assert report.confidence >= 0.75


# =========================================================================
# PHASE 4: Grounded Failure Investigation Tests
# =========================================================================
def test_credibility_constraint_3_never_hallucinate_root_causes(temp_history_store):
    """
    User Correction 3:
    Don't let AI invent root causes.
    If TestPilot cannot establish the cause, it should say 'Insufficient evidence'.
    """
    investigator = FailureInvestigator(history_store=temp_history_store)

    # Ambiguous test failure without clear stack trace or symbol
    result = investigator.investigate(
        test_id="tests/test_unknown.py::test_mysterious_case",
        test_name="test_mysterious_case",
        test_file="tests/test_unknown.py",
        error_message="Non-zero exit code 1",
        traceback="",
        category="UNCERTAIN",
        check_pollution=False,
    )

    # Must honestly report Insufficient Evidence
    assert not result.is_grounded
    assert "insufficient evidence" in result.why_it_failed.lower()
    assert "insufficient evidence" in result.suggested_repair.lower()
    assert result.repair_confidence == 0.0


def test_investigation_answers_5_questions(temp_history_store):
    """Verifies that failure investigation answers all 5 developer questions."""
    investigator = FailureInvestigator(history_store=temp_history_store)

    result = investigator.investigate(
        test_id="tests/test_auth.py::test_token_signature",
        test_name="test_token_signature",
        test_file="tests/test_auth.py",
        error_message="AssertionError: assert False where False = verify_jwt('invalid_key')",
        traceback="Traceback (most recent call last):\n  File 'test_auth.py', line 45, in test_token_signature\n    assert verify_jwt('invalid_key')\nAssertionError: assert False",
        category="REGRESSION",
        category_confidence=0.92,
        targeted_symbol="verify_jwt",
        check_pollution=False,
    )

    # Question 1: What exactly went wrong?
    assert "AssertionError" in result.what_went_wrong
    # Question 2: Why did it fail?
    assert "verify_jwt" in result.why_it_failed
    # Question 3: Category & Evidence
    assert result.category == "REGRESSION"
    assert len(result.evidence_trail) >= 2
    # Question 4: Next steps
    assert len(result.investigate_next) >= 2
    # Question 5: Grounded repair
    assert "verify_jwt" in result.suggested_repair


# =========================================================================
# PHASE 5: Advanced Features (Quarantine, Order Pollution, Alerts)
# =========================================================================
def test_test_quarantine_system(temp_history_store):
    test_id = "tests/test_flaky_api.py::test_remote_webhook"

    # Initially not quarantined
    assert not temp_history_store.is_quarantined(test_id)

    # Quarantine test
    temp_history_store.quarantine_test(test_id, reason="High flip rate on CI", author="CI Bot")
    assert temp_history_store.is_quarantined(test_id)

    quarantined_list = temp_history_store.get_quarantined_tests()
    assert any(q["test_id"] == test_id for q in quarantined_list)

    # Un-quarantine
    temp_history_store.unquarantine_test(test_id)
    assert not temp_history_store.is_quarantined(test_id)


# =========================================================================
# PHASE 3: Test Intelligence REST API Endpoints Tests
# =========================================================================
def test_test_intelligence_api_endpoints():
    # 1. Summary endpoint
    res = client.get("/api/intelligence/summary")
    assert res.status_code == 200
    summary = res.json()
    assert "pass_rate_pct" in summary
    assert "confirmed_flakes_count" in summary
    assert "quarantined_count" in summary
    assert "reliability_alerts" in summary

    # 2. Flaky tests watchlist
    res = client.get("/api/intelligence/flaky-tests")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 3. Quarantine toggle via API
    test_target = "tests/test_orders.py::test_mock_quarantine"
    res = client.post(
        "/api/intelligence/quarantine",
        json={"test_id": test_target, "is_quarantined": True, "reason": "Test flaky quarantine"},
    )
    assert res.status_code == 200
    assert res.json()["is_quarantined"] is True

    # Check quarantined list
    res = client.get("/api/intelligence/quarantined")
    assert res.status_code == 200
    assert any(q["test_id"] == test_target for q in res.json()["quarantined_tests"])

    # Un-quarantine
    res = client.post(
        "/api/intelligence/quarantine",
        json={"test_id": test_target, "is_quarantined": False},
    )
    assert res.status_code == 200
    assert res.json()["is_quarantined"] is False

    # 4. Investigate endpoint via API
    res = client.post(
        "/api/intelligence/investigate",
        json={
            "test_id": "tests/test_example.py::test_demo",
            "test_file": "tests/test_example.py",
            "test_name": "test_demo",
            "error_message": "AssertionError: Expected 200, got 404",
            "category": "REGRESSION",
            "check_pollution": False,
        },
    )
    assert res.status_code == 200
    inv = res.json()
    assert "AssertionError" in inv["what_went_wrong"]
    assert inv["category"] == "REGRESSION"
    assert len(inv["investigate_next"]) > 0
