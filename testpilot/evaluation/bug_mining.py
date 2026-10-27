"""
Historical Bug-Fix Ground Truth Mining Subsystem for TestPilot.
Mines candidate bug-fix commits from Git repository history, identifies parent (buggy)
and child (fixed) revisions, extracts candidate regression tests, and validates before/after
outcomes to construct independently proven regression benchmark ground truth.
"""

from __future__ import annotations

import ast
import datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional
import uuid

from testpilot.ast_engine.treesitter_parser import ASTDiffParser
from testpilot.evaluation.custom_models import (
    BenchmarkSourceType,
    CustomBenchmarkCase,
    GroundTruthReviewStatus,
    GroundTruthValidationLevel,
)
from testpilot.evaluation.generation_models import HistoricalBugCandidate
from testpilot.evaluation.models import GroundTruth
from testpilot.evolution.models import ChangedSymbol, ChangeType

logger = logging.getLogger(__name__)


class HistoricalBugMiner:
    """
    Derives candidate regression ground truth from real Git history:
    1. Scans commit log for bug-fix indicators and associated test modifications.
    2. Reconstructs parent (buggy) and target (fixed) revisions.
    3. Executes candidate tests against both revisions.
    4. Certifies a test as verified regression ground truth ONLY if it reproducibly
       fails on the buggy revision and passes on the fixed revision.
    """

    BUG_COMMIT_PATTERNS = [
        re.compile(r"\b(?:fix(?:es|ed|ing)?|bug(?:s)?|defect|issue|patch|resolve(?:s|d)?)\b", re.IGNORECASE),
        re.compile(r"#\d+", re.IGNORECASE),
    ]

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Path is not a Git repository: {self.repo_path}")

    def mine_candidate_bug_fixes(
        self,
        repo_id: str,
        max_commits: int = 50,
        max_candidates: int = 10,
    ) -> list[HistoricalBugCandidate]:
        """
        Discovers candidate bug-fix commits in repository Git history that
        both modified implementation code and added or modified test files.
        """
        # Read git log with hashes and subject
        cmd = [
            "git",
            "log",
            f"-n{max_commits}",
            "--pretty=format:%H|%P|%s",
            "--no-merges",
        ]
        res = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            logger.warning(f"Failed to query git log: {res.stderr}")
            return []

        candidates: list[HistoricalBugCandidate] = []

        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split("|", 2)
            if len(parts) < 3:
                continue
            commit_sha, parents, subject = parts[0], parts[1], parts[2]
            parent_shas = parents.split()
            if not parent_shas:
                continue
            parent_sha = parent_shas[0]

            # Check if commit message has bug-fix indicators
            is_bug_message = any(p.search(subject) for p in self.BUG_COMMIT_PATTERNS)
            if not is_bug_message:
                continue

            # Inspect changed files in this commit
            files_cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha]
            files_res = subprocess.run(
                files_cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if files_res.returncode != 0:
                continue

            changed_files = [f.strip() for f in files_res.stdout.splitlines() if f.strip()]
            test_files = [f for f in changed_files if self._is_test_file(f)]
            source_files = [f for f in changed_files if not self._is_test_file(f) and f.endswith(".py")]

            # We need at least one source file AND at least one test file
            if not source_files or not test_files:
                continue

            # Extract test function names from test files at commit
            candidate_tests = self._extract_tests_at_commit(commit_sha, test_files)
            if not candidate_tests:
                continue

            # Extract changed symbols
            changed_symbols = self._extract_changed_symbols_between(parent_sha, commit_sha, source_files)

            cand_id = f"bug_{repo_id}_{commit_sha[:8]}"
            candidate = HistoricalBugCandidate(
                candidate_id=cand_id,
                repo_id=repo_id,
                commit_sha=commit_sha,
                parent_commit_sha=parent_sha,
                commit_message=subject,
                changed_files=source_files,
                changed_symbols=changed_symbols,
                test_files=test_files,
                candidate_tests=candidate_tests,
                validation_status=GroundTruthReviewStatus.CANDIDATE,
                validation_level=GroundTruthValidationLevel.PROPOSED_AWAITING_REVIEW,
                confidence=0.6,
                evidence_notes=f"Identified from commit '{subject}' with {len(test_files)} test files modified.",
            )
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break

        return candidates

    def validate_bug_candidate(
        self,
        candidate: HistoricalBugCandidate,
        timeout_sec: int = 30,
    ) -> HistoricalBugCandidate:
        """
        Validates whether candidate tests reproduce the historical defect:
        1. Materializes workspace at parent_commit (buggy) -> executes tests -> verifies FAIL.
        2. Materializes workspace at commit_sha (fixed) -> executes tests -> verifies PASS.
        3. Updates candidate status and confidence.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="tp_bug_val_"))
        try:
            ws_buggy = temp_dir / "buggy"
            ws_fixed = temp_dir / "fixed"
            ws_buggy.mkdir(parents=True, exist_ok=True)
            ws_fixed.mkdir(parents=True, exist_ok=True)

            # Export buggy revision archive
            self._export_archive(candidate.parent_commit_sha, ws_buggy)
            # Copy over new test files if the test was newly created in the fix
            # so the buggy revision can actually be evaluated against the new test
            self._overlay_tests_from_commit(candidate.commit_sha, candidate.test_files, ws_buggy)

            # Export fixed revision archive
            self._export_archive(candidate.commit_sha, ws_fixed)

            # Run tests on buggy version
            buggy_results = self._run_tests_in_workspace(ws_buggy, candidate.candidate_tests, timeout_sec)
            # Run tests on fixed version
            fixed_results = self._run_tests_in_workspace(ws_fixed, candidate.candidate_tests, timeout_sec)

            candidate.before_result = buggy_results.get("summary", "UNKNOWN")
            candidate.after_result = fixed_results.get("summary", "UNKNOWN")

            # True bug detection: Fails on buggy due to assertion defect (NOT environment), Passes on fixed
            buggy_is_env = buggy_results.get("is_env_error", False) or buggy_results.get("summary") in ("ENVIRONMENT_ERROR", "RUNNER_ERROR", "TIMEOUT", "ERROR")
            fixed_passed = fixed_results.get("all_passed", False)
            buggy_defect_failed = (buggy_results.get("summary") == "FAILED" or buggy_results.get("has_failure", False)) and not buggy_is_env

            before_out = (buggy_results.get("output") or "")[:2000]
            after_out = (fixed_results.get("output") or "")[:2000]

            if buggy_defect_failed and fixed_passed:
                candidate.validation_status = GroundTruthReviewStatus.VALIDATED
                candidate.validation_level = GroundTruthValidationLevel.HISTORICAL_REGRESSION_VERIFIED
                candidate.confidence = 0.95
                candidate.evidence_notes = (
                    f"Verified regression detector: assertion defect reproduced on buggy commit {candidate.parent_commit_sha[:8]} "
                    f"and resolved on fixed commit {candidate.commit_sha[:8]}.\n"
                    f"--- Before Fix Output ---\n{before_out}\n"
                    f"--- After Fix Output ---\n{after_out}"
                )
            elif buggy_is_env:
                candidate.validation_status = GroundTruthReviewStatus.REJECTED
                candidate.validation_level = GroundTruthValidationLevel.INSUFFICIENT_EVIDENCE
                candidate.confidence = 0.0
                candidate.evidence_notes = f"Rejected: Buggy revision failed due to environment, missing import, or fixture problem ({candidate.before_result}) rather than defect assertion."
            elif not buggy_results.get("has_failure", False) and fixed_passed:
                candidate.validation_status = GroundTruthReviewStatus.REJECTED
                candidate.validation_level = GroundTruthValidationLevel.INSUFFICIENT_EVIDENCE
                candidate.confidence = 0.2
                candidate.evidence_notes = "Rejected: Test passes on both buggy and fixed commits (not regression-sensitive)."
            elif buggy_defect_failed and not fixed_passed:
                candidate.validation_status = GroundTruthReviewStatus.REJECTED
                candidate.validation_level = GroundTruthValidationLevel.INSUFFICIENT_EVIDENCE
                candidate.confidence = 0.1
                candidate.evidence_notes = "Rejected: Test fails on both commits (persistent defect or broken assertion)."
            else:
                candidate.validation_status = GroundTruthReviewStatus.NEEDS_HUMAN_REVIEW
                candidate.validation_level = GroundTruthValidationLevel.PROPOSED_AWAITING_REVIEW
                candidate.confidence = 0.4
                candidate.evidence_notes = f"Inconclusive execution outcomes across commits: before={candidate.before_result}, after={candidate.after_result}."

            return candidate

        except Exception as e:
            logger.error(f"Failed to validate bug candidate {candidate.candidate_id}: {e}")
            candidate.validation_status = GroundTruthReviewStatus.REJECTED
            candidate.evidence_notes = f"Validation failed with error: {str(e)}"
            return candidate
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def convert_candidate_to_benchmark_case(
        self,
        candidate: HistoricalBugCandidate,
    ) -> CustomBenchmarkCase:
        """Converts a validated bug candidate into a versioned CustomBenchmarkCase."""
        expected_callers = candidate.candidate_tests
        case_id = f"{candidate.repo_id}_hist_{candidate.commit_sha[:8]}"

        gt = GroundTruth(
            case_id=case_id,
            expected_changed_symbols=candidate.changed_symbols,
            expected_impacted_symbols=candidate.changed_symbols,
            expected_test_callers=expected_callers,
            provenance=f"Historical bug fix verified across {candidate.parent_commit_sha[:8]}..{candidate.commit_sha[:8]}",
            notes=candidate.commit_message,
            is_verified=(candidate.validation_status == GroundTruthReviewStatus.VALIDATED),
        )

        return CustomBenchmarkCase(
            case_id=case_id,
            name=f"Fix {candidate.commit_sha[:8]}: {candidate.commit_message[:50]}",
            description=candidate.commit_message,
            source_type=BenchmarkSourceType.HISTORICAL_BUG_FIX,
            base_commit=candidate.parent_commit_sha,
            target_commit=candidate.commit_sha,
            commit_sha=candidate.commit_sha,
            ground_truth=gt,
            validation_level=candidate.validation_level,
            review_status=candidate.validation_status,
            validation_evidence={
                "before_result": candidate.before_result,
                "after_result": candidate.after_result,
                "confidence": candidate.confidence,
                "evidence_notes": candidate.evidence_notes,
            },
            mutant_kill_rate=1.0 if candidate.validation_status == GroundTruthReviewStatus.VALIDATED else 0.0,
            fault_detection_rate=1.0 if candidate.validation_status == GroundTruthReviewStatus.VALIDATED else 0.0,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    @staticmethod
    def _is_test_file(path_str: str) -> bool:
        p = path_str.lower().replace("\\", "/")
        return (
            (p.startswith("tests/") or "/tests/" in p or p.startswith("test/"))
            and (p.endswith(".py"))
            and ("test_" in p or "_test" in p)
        )

    def _export_archive(self, commit_sha: str, dest_dir: Path) -> None:
        cmd = ["git", "archive", commit_sha]
        proc = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, check=True)
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
            tf.extractall(path=dest_dir)

    def _overlay_tests_from_commit(self, commit_sha: str, test_files: list[str], dest_dir: Path) -> None:
        for tf in test_files:
            cmd = ["git", "show", f"{commit_sha}:{tf}"]
            proc = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout:
                out_path = dest_dir / tf
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(proc.stdout, encoding="utf-8")

    def _extract_tests_at_commit(self, commit_sha: str, test_files: list[str]) -> list[str]:
        test_ids: list[str] = []
        for tf in test_files:
            cmd = ["git", "show", f"{commit_sha}:{tf}"]
            proc = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                continue
            try:
                tree = ast.parse(proc.stdout, filename=tf)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and (node.name.startswith("test_") or node.name.endswith("_test")):
                        norm_tf = tf.replace("\\", "/")
                        test_ids.append(f"{norm_tf}::{node.name}")
            except Exception:
                continue
        return test_ids

    def _extract_changed_symbols_between(self, base_commit: str, target_commit: str, files: list[str]) -> list[str]:
        syms: set[str] = set()
        diff_cmd = ["git", "diff", "-U0", base_commit, target_commit, "--"] + files
        proc = subprocess.run(diff_cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("@@"):
                    # Extract function name from hunk header e.g. @@ -10,0 +11,5 @@ def my_func():
                    m = re.search(r"def\s+([a-zA-Z0-9_]+)", line)
                    if m:
                        syms.add(m.group(1))
                    m2 = re.search(r"class\s+([a-zA-Z0-9_]+)", line)
                    if m2:
                        syms.add(m2.group(1))
        return sorted(syms)

    def _run_tests_in_workspace(self, ws: Path, test_ids: list[str], timeout_sec: int) -> dict[str, Any]:
        if not test_ids:
            return {"summary": "NO_TESTS", "has_failure": False, "all_passed": False}

        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + test_ids
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(ws)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if proc.returncode == 0:
                return {"summary": "PASSED", "has_failure": False, "all_passed": True, "output": stdout}
            elif proc.returncode == 1:
                # Distinguish test assertion failures from environment/setup/import defects
                combined = stdout + "\n" + stderr
                is_env = any(marker in combined for marker in [
                    "ModuleNotFoundError", "ImportError", "fixture '", "ERROR collecting", "CollectionError", "ScopeMismatch"
                ])
                if is_env:
                    return {"summary": "ENVIRONMENT_ERROR", "has_failure": True, "is_env_error": True, "all_passed": False, "output": combined}
                return {"summary": "FAILED", "has_failure": True, "is_env_error": False, "all_passed": False, "output": stdout}
            else:
                return {"summary": "RUNNER_ERROR", "has_failure": True, "is_env_error": True, "all_passed": False, "output": stderr or stdout}
        except subprocess.TimeoutExpired:
            return {"summary": "TIMEOUT", "has_failure": True, "is_env_error": True, "all_passed": False, "output": "Timed out"}
        except Exception as e:
            return {"summary": "RUNNER_ERROR", "has_failure": True, "is_env_error": True, "all_passed": False, "output": str(e)}
