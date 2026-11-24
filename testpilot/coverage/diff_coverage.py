"""
Revision Comparison Engine for TestPilot Test Coverage Intelligence.
Compares runtime line and function coverage between Git revisions, detects
coverage regressions, identifies untested changed code, and handles file additions,
modifications, deletions, and missing baseline reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Optional

from testpilot.coverage.models import (
    CoverageRegression,
    CoverageReport,
    FileDiffCoverage,
    FunctionCoverage,
    FunctionStatus,
    RevisionComparisonReport,
)
from testpilot.coverage.storage import CoverageStore

logger = logging.getLogger("testpilot.coverage.diff_coverage")


class RevisionCoverageComparator:
    """
    Compares coverage telemetry across Git revisions or execution runs.
    Accurately detects coverage regressions, newly covered code, and uncovered modifications.
    """

    def __init__(self, store: Optional[CoverageStore] = None) -> None:
        self.store = store or CoverageStore.get_instance()

    def compare_commits(
        self,
        repo_id: str,
        base_commit: Optional[str],
        target_commit: str,
        repo_path: Optional[str | Path] = None,
    ) -> RevisionComparisonReport:
        """
        Loads coverage reports for base and target commits from CoverageStore and performs comparison.
        Gracefully handles missing base coverage data.
        """
        target_report = self.store.get_latest_report(repo_id, commit_hash=target_commit)
        if not target_report:
            # Fallback: maybe target_commit was passed as a coverage_run_id
            target_report = self.store.get_report(target_commit)

        if not target_report:
            # Try getting any latest report for repo_id
            target_report = self.store.get_latest_report(repo_id)

        if not target_report:
            # Empty fallback when target report does not exist
            return RevisionComparisonReport(
                base_commit=base_commit or "unknown",
                target_commit=target_commit,
                base_available=False,
                notes=[f"Target coverage report for commit or run '{target_commit}' was not found in store."],
            )

        base_report: Optional[CoverageReport] = None
        if base_commit and base_commit != target_commit:
            base_report = self.store.get_latest_report(repo_id, commit_hash=base_commit)
            if not base_report:
                base_report = self.store.get_report(base_commit)

        return self.compare_reports(
            base_report=base_report,
            target_report=target_report,
            repo_path=repo_path or target_report.repo_path,
        )

    def compare_reports(
        self,
        base_report: Optional[CoverageReport],
        target_report: CoverageReport,
        repo_path: Optional[str | Path] = None,
    ) -> RevisionComparisonReport:
        """
        Core comparator logic comparing two CoverageReport objects.
        """
        # Case 1: Base report is unavailable
        if base_report is None:
            return self._build_single_report_comparison(target_report)

        base_commit = base_report.commit_hash
        target_commit = target_report.commit_hash

        # Calculate high-level deltas
        line_delta = round(target_report.line_coverage_pct - base_report.line_coverage_pct, 2)
        func_delta = round(target_report.function_coverage_pct - base_report.function_coverage_pct, 2)

        # Detect git-level file changes if repository path is valid
        git_diff_status = self._get_git_changed_files(repo_path, base_commit, target_commit)

        # Build function lookups: (file_path, qualified_name) -> FunctionCoverage
        base_funcs: dict[tuple[str, str], FunctionCoverage] = {}
        for fs in base_report.file_summaries:
            for fn in fs.functions:
                base_funcs[(fs.file_path, fn.qualified_name)] = fn

        target_funcs: dict[tuple[str, str], FunctionCoverage] = {}
        for fs in target_report.file_summaries:
            for fn in fs.functions:
                target_funcs[(fs.file_path, fn.qualified_name)] = fn

        regressions: list[CoverageRegression] = []
        newly_covered: list[str] = []
        newly_uncovered: list[str] = []
        uncovered_new_or_modified: list[str] = []

        # Analyze regressions and function state shifts
        for (fpath, qname), t_fn in target_funcs.items():
            b_fn = base_funcs.get((fpath, qname))
            if b_fn is not None:
                # Existed in both revisions
                # Regression check: coverage dropped or became uncovered
                is_regressed = False
                lines_lost: list[int] = []

                if b_fn.coverage_pct > t_fn.coverage_pct and t_fn.coverage_pct < 100.0:
                    is_regressed = True
                    # Lines executed in base that are missing in target
                    base_executed = set(b_fn.executed_lines)
                    target_executed = set(t_fn.executed_lines)
                    lines_lost = sorted(list(base_executed - target_executed))

                if is_regressed or (b_fn.status != FunctionStatus.UNCOVERED and t_fn.status == FunctionStatus.UNCOVERED):
                    regressions.append(
                        CoverageRegression(
                            file_path=fpath,
                            function_name=qname,
                            previous_coverage_pct=b_fn.coverage_pct,
                            current_coverage_pct=t_fn.coverage_pct,
                            previous_status=b_fn.status,
                            current_status=t_fn.status,
                            lines_lost=lines_lost,
                        )
                    )

                # Newly covered check
                if b_fn.status == FunctionStatus.UNCOVERED and t_fn.status != FunctionStatus.UNCOVERED:
                    newly_covered.append(f"{fpath}::{qname}")

                # Newly uncovered check
                if b_fn.status != FunctionStatus.UNCOVERED and t_fn.status == FunctionStatus.UNCOVERED:
                    newly_uncovered.append(f"{fpath}::{qname}")

                # Modified function with zero coverage
                if git_diff_status.get(fpath) in ("modified", "added") and t_fn.status == FunctionStatus.UNCOVERED:
                    uncovered_new_or_modified.append(f"{fpath}::{qname}")
            else:
                # Function is newly added in target revision
                if t_fn.status != FunctionStatus.UNCOVERED and t_fn.coverage_pct > 0:
                    newly_covered.append(f"{fpath}::{qname}")
                else:
                    uncovered_new_or_modified.append(f"{fpath}::{qname}")

        # Per-file diff coverage computation
        all_files = sorted(set(list(base_report.files.keys()) + list(target_report.files.keys())))
        base_summaries_map = {fs.file_path: fs for fs in base_report.file_summaries}
        target_summaries_map = {fs.file_path: fs for fs in target_report.file_summaries}

        file_diffs: list[FileDiffCoverage] = []
        for fpath in all_files:
            b_line = base_report.files.get(fpath)
            t_line = target_report.files.get(fpath)
            b_summary = base_summaries_map.get(fpath)
            t_summary = target_summaries_map.get(fpath)

            change_type = git_diff_status.get(fpath)
            if not change_type:
                if b_line is None and t_line is not None:
                    change_type = "added"
                elif b_line is not None and t_line is None:
                    change_type = "deleted"
                else:
                    change_type = "modified"

            b_cov = b_line.line_coverage_pct if b_line else None
            t_cov = t_line.line_coverage_pct if t_line else None
            line_cov_delta = round(t_cov - b_cov, 2) if (b_cov is not None and t_cov is not None) else None

            b_fcov = b_summary.function_coverage_pct if b_summary else None
            t_fcov = t_summary.function_coverage_pct if t_summary else None
            func_cov_delta = round(t_fcov - b_fcov, 2) if (b_fcov is not None and t_fcov is not None) else None

            file_new_cov: list[str] = [fn for fn in newly_covered if fn.startswith(f"{fpath}::")]
            file_new_uncov: list[str] = [fn for fn in newly_uncovered if fn.startswith(f"{fpath}::")]
            uncovered_changed = t_line.missing_lines if t_line else []

            file_diffs.append(
                FileDiffCoverage(
                    file_path=fpath,
                    change_type=change_type,
                    base_line_coverage_pct=b_cov,
                    target_line_coverage_pct=t_cov,
                    line_coverage_delta=line_cov_delta,
                    base_function_coverage_pct=b_fcov,
                    target_function_coverage_pct=t_fcov,
                    function_coverage_delta=func_cov_delta,
                    newly_covered_functions=file_new_cov,
                    newly_uncovered_functions=file_new_uncov,
                    uncovered_changed_lines=uncovered_changed,
                )
            )

        notes: list[str] = []
        if regressions:
            notes.append(f"Detected {len(regressions)} coverage regression(s) where functions lost test coverage.")
        if uncovered_new_or_modified:
            notes.append(f"Identified {len(uncovered_new_or_modified)} added or modified function(s) lacking test coverage.")
        if line_delta > 0:
            notes.append(f"Overall line coverage increased by +{line_delta}%.")
        elif line_delta < 0:
            notes.append(f"Overall line coverage decreased by {line_delta}%.")

        return RevisionComparisonReport(
            base_commit=base_commit,
            target_commit=target_commit,
            base_run_id=base_report.coverage_run_id,
            target_run_id=target_report.coverage_run_id,
            base_available=True,
            line_coverage_delta=line_delta,
            function_coverage_delta=func_delta,
            base_line_coverage_pct=base_report.line_coverage_pct,
            target_line_coverage_pct=target_report.line_coverage_pct,
            base_function_coverage_pct=base_report.function_coverage_pct,
            target_function_coverage_pct=target_report.function_coverage_pct,
            regressions=regressions,
            newly_covered_functions=newly_covered,
            newly_uncovered_functions=newly_uncovered,
            uncovered_new_or_modified_functions=uncovered_new_or_modified,
            files=file_diffs,
            notes=notes,
        )

    def _build_single_report_comparison(self, target_report: CoverageReport) -> RevisionComparisonReport:
        """Handles the case where base revision coverage data is unavailable."""
        uncovered: list[str] = []
        for fs in target_report.file_summaries:
            for fn in fs.functions:
                if fn.status == FunctionStatus.UNCOVERED:
                    uncovered.append(f"{fs.file_path}::{fn.qualified_name}")

        file_diffs: list[FileDiffCoverage] = []
        for fs in target_report.file_summaries:
            t_line = target_report.files.get(fs.file_path)
            file_diffs.append(
                FileDiffCoverage(
                    file_path=fs.file_path,
                    change_type="existing",
                    base_line_coverage_pct=None,
                    target_line_coverage_pct=fs.line_coverage_pct,
                    line_coverage_delta=None,
                    base_function_coverage_pct=None,
                    target_function_coverage_pct=fs.function_coverage_pct,
                    function_coverage_delta=None,
                    newly_covered_functions=[],
                    newly_uncovered_functions=[],
                    uncovered_changed_lines=t_line.missing_lines if t_line else [],
                )
            )

        return RevisionComparisonReport(
            base_commit="none",
            target_commit=target_report.commit_hash,
            base_run_id=None,
            target_run_id=target_report.coverage_run_id,
            base_available=False,
            line_coverage_delta=0.0,
            function_coverage_delta=0.0,
            base_line_coverage_pct=0.0,
            target_line_coverage_pct=target_report.line_coverage_pct,
            base_function_coverage_pct=0.0,
            target_function_coverage_pct=target_report.function_coverage_pct,
            regressions=[],
            newly_covered_functions=[],
            newly_uncovered_functions=[],
            uncovered_new_or_modified_functions=uncovered,
            files=file_diffs,
            notes=[
                "Base revision coverage report is unavailable. Baseline coverage metrics are shown for target revision.",
                f"Repository target commit '{target_report.commit_hash}' contains {len(uncovered)} uncovered functions.",
            ],
        )

    def _get_git_changed_files(
        self,
        repo_path: Optional[str | Path],
        base_commit: str,
        target_commit: str,
    ) -> dict[str, str]:
        """
        Executes git diff --name-status to identify added/modified/deleted files between revisions.
        Returns dict mapping normalized relative file_path -> change_type.
        """
        if not repo_path or base_commit == target_commit or base_commit == "none":
            return {}

        path_obj = Path(repo_path).resolve()
        if not (path_obj / ".git").exists():
            return {}

        try:
            cmd = ["git", "diff", "--name-status", base_commit, target_commit]
            result = subprocess.run(
                cmd,
                cwd=str(path_obj),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug(f"Git diff exited with code {result.returncode}: {result.stderr}")
                return {}

            status_map = {
                "A": "added",
                "M": "modified",
                "D": "deleted",
                "R": "renamed",
                "C": "copied",
            }
            res_dict: dict[str, str] = {}
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split("\t")
                if not parts:
                    continue
                code = parts[0][0]
                fname = parts[-1].replace("\\", "/")
                res_dict[fname] = status_map.get(code, "modified")
            return res_dict
        except Exception as e:
            logger.debug(f"Could not retrieve git changed files between {base_commit} and {target_commit}: {e}")
            return {}

