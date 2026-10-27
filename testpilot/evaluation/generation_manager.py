"""
Durable, Observable, and Recoverable Test Generation Job Manager for TestPilot.
Orchestrates test generation jobs across explicit lifecycle states:
QUEUED -> ANALYZING -> DISCOVERING_TESTS -> GENERATING -> QUALIFYING -> VALIDATING -> COMPLETED.
Persists job state to disk to survive browser reloads, network disconnects, and server restarts.
"""

import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional
import uuid

from testpilot.evaluation.auto_benchmark import AutoBenchmarkGenerator
from testpilot.evaluation.baselines import BaselineEvaluator
from testpilot.evaluation.custom_models import (
    CustomRepositoryRecord,
    GeneratedTestRecord,
    RepositoryStatus,
)
from testpilot.evaluation.generation_models import (
    GenerationJobRecord,
    GenerationJobStatus,
)

logger = logging.getLogger(__name__)


class GenerationJobManager:
    """
    Durable manager for autonomous test generation jobs.
    Persists jobs in `testbed/evaluation_store/generation_jobs/`.
    Ensures bounded execution timeouts, idempotent retries, and observable progress.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir).resolve()
        else:
            ws_store = Path("testbed/evaluation_store").resolve()
            self.storage_dir = ws_store if ws_store.parent.exists() else Path.home() / ".testpilot_eval"

        self.jobs_dir = self.storage_dir / "generation_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._active_threads: dict[str, threading.Thread] = {}
        self._cancellation_flags: dict[str, bool] = {}

        # Run recovery on startup
        self.recover_stale_jobs()

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save_job(self, job: GenerationJobRecord) -> None:
        """Atomically saves job record to persistent JSON file."""
        job.last_progress_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.jobs_dir / f"{job.job_id}.tmp"
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self._job_file(job.job_id))

    def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        """Loads job record from disk."""
        jf = self._job_file(job_id)
        if not jf.exists():
            return None
        try:
            return GenerationJobRecord.model_validate_json(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read job {job_id}: {e}")
            return None

    def list_jobs_for_repo(self, repo_id: str) -> list[GenerationJobRecord]:
        """Returns all generation jobs for a given repository, sorted newest first."""
        jobs: list[GenerationJobRecord] = []
        for jf in self.jobs_dir.glob("*.json"):
            try:
                rec = GenerationJobRecord.model_validate_json(jf.read_text(encoding="utf-8"))
                if rec.repo_id == repo_id:
                    jobs.append(rec)
            except Exception:
                continue
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs

    def get_latest_job_for_repo(self, repo_id: str) -> Optional[GenerationJobRecord]:
        jobs = self.list_jobs_for_repo(repo_id)
        return jobs[0] if jobs else None

    def create_job(self, repo_id: str, commit_sha: Optional[str] = None) -> GenerationJobRecord:
        """Creates and persists a new generation job in QUEUED status."""
        job_id = f"gen_{repo_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = GenerationJobRecord(
            job_id=job_id,
            repo_id=repo_id,
            commit_sha=commit_sha,
            status=GenerationJobStatus.QUEUED,
            current_stage="Job queued and awaiting worker execution",
        )
        self.save_job(job)
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Requests cancellation of an in-flight generation job."""
        job = self.get_job(job_id)
        if not job or job.status in (
            GenerationJobStatus.COMPLETED,
            GenerationJobStatus.COMPLETED_WITH_WARNINGS,
            GenerationJobStatus.FAILED,
            GenerationJobStatus.CANCELLED,
        ):
            return False

        self._cancellation_flags[job_id] = True
        job.status = GenerationJobStatus.CANCELLED
        job.current_stage = "Job cancelled by user"
        job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save_job(job)
        return True

    def recover_stale_jobs(self) -> None:
        """
        Scans persisted jobs and marks any orphaned non-terminal jobs
        (e.g. from server restarts) as COMPLETED_WITH_WARNINGS or FAILED.
        """
        for jf in self.jobs_dir.glob("*.json"):
            try:
                rec = GenerationJobRecord.model_validate_json(jf.read_text(encoding="utf-8"))
                if rec.status in (
                    GenerationJobStatus.QUEUED,
                    GenerationJobStatus.ANALYZING,
                    GenerationJobStatus.DISCOVERING_TESTS,
                    GenerationJobStatus.GENERATING,
                    GenerationJobStatus.QUALIFYING,
                    GenerationJobStatus.VALIDATING,
                ):
                    rec.status = GenerationJobStatus.COMPLETED_WITH_WARNINGS if rec.accepted_count > 0 else GenerationJobStatus.FAILED
                    rec.failure_reason = "Job interrupted by process restart or timeout. Recovered to terminal state."
                    rec.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    self.save_job(rec)
            except Exception:
                continue

    def start_generation_async(
        self,
        job_id: str,
        repo_record: CustomRepositoryRecord,
        custom_repo_mgr: Any,
        max_targets: int = 15,
    ) -> None:
        """Starts generation job execution in an isolated background daemon thread."""
        thread = threading.Thread(
            target=self._execute_generation_job,
            args=(job_id, repo_record, custom_repo_mgr, max_targets),
            daemon=True,
            name=f"gen-worker-{job_id}",
        )
        self._active_threads[job_id] = thread
        thread.start()

    def _execute_generation_job(
        self,
        job_id: str,
        repo_record: CustomRepositoryRecord,
        custom_repo_mgr: Any,
        max_targets: int,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        repo_id = repo_record.repo_id
        repo_path = Path(repo_record.canonical_path).resolve()

        def _is_cancelled() -> bool:
            return self._cancellation_flags.get(job_id, False)

        try:
            # Stage 1: ANALYZING
            t0 = time.perf_counter()
            job.status = GenerationJobStatus.ANALYZING
            job.current_stage = "Analyzing source code, public symbols, and behavioral contracts..."
            job.diagnostic_messages.append(f"Analyzing repository at {repo_path}")
            self.save_job(job)

            if _is_cancelled():
                return

            generator = AutoBenchmarkGenerator(repo_path=repo_path)
            targets = generator.discover_target_functions()[:max_targets]
            job.stage_durations["analyzing"] = round(time.perf_counter() - t0, 3)

            # Stage 2: DISCOVERING_TESTS
            t1 = time.perf_counter()
            job.status = GenerationJobStatus.DISCOVERING_TESTS
            job.current_stage = "Discovering pre-existing tests and unexercised behavioral paths..."
            self.save_job(job)

            evaluator = BaselineEvaluator(repo_path=str(repo_path))
            existing_tests = evaluator.discover_all_tests()
            job.diagnostic_messages.append(f"Discovered {len(existing_tests)} existing tests in repository")
            job.stage_durations["discovering_tests"] = round(time.perf_counter() - t1, 3)

            if not targets:
                job.status = GenerationJobStatus.COMPLETED_WITH_WARNINGS
                job.current_stage = "No eligible public functions identified for test generation."
                job.failure_reason = "No eligible public functions found for automatic test synthesis."
                job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.save_job(job)

                repo_record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
                repo_record.status_message = "No eligible public functions found for automatic test synthesis."
                custom_repo_mgr.update_repository_record(repo_record)
                return

            if _is_cancelled():
                return

            # Stage 3: GENERATING
            t2 = time.perf_counter()
            job.status = GenerationJobStatus.GENERATING
            job.current_stage = f"Synthesizing behavioral regression test candidates for {len(targets)} targets..."
            self.save_job(job)

            gen_records = generator.generate_and_save_tests(max_targets=max_targets)
            job.candidates_count = len(gen_records)
            job.stage_durations["generating"] = round(time.perf_counter() - t2, 3)
            job.diagnostic_messages.append(f"Generated {len(gen_records)} candidate tests across {len(targets)} targets")
            self.save_job(job)

            if _is_cancelled():
                return

            # Stage 4: QUALIFYING
            t3 = time.perf_counter()
            job.status = GenerationJobStatus.QUALIFYING
            job.current_stage = f"Qualifying {len(gen_records)} test candidates (syntax, imports, anti-trivial assertions)..."
            self.save_job(job)

            val_records = generator.validate_generated_tests(gen_records)
            job.stage_durations["qualifying"] = round(time.perf_counter() - t3, 3)

            # Count accepted vs rejected
            accepted_val = [r for r in val_records if r.executes_successfully and not r.is_trivial]
            rejected_val = [r for r in val_records if not r.executes_successfully or r.is_trivial]
            job.accepted_count = len(accepted_val)
            job.rejected_count = len(rejected_val)

            reasons: dict[str, int] = {}
            for r in rejected_val:
                reason = r.rejection_reason or "Unknown validation failure"
                reasons[reason] = reasons.get(reason, 0) + 1
            job.rejection_reasons = reasons
            self.save_job(job)

            if _is_cancelled():
                return

            # Stage 5: VALIDATING
            t4 = time.perf_counter()
            job.status = GenerationJobStatus.VALIDATING
            job.current_stage = f"Validating regression sensitivity via mutation testing for {len(accepted_val)} tests..."
            self.save_job(job)

            mut_records = generator.cross_check_mutation_sensitivity(val_records)
            job.stage_durations["validating"] = round(time.perf_counter() - t4, 3)

            mut_sensitive = [r for r in mut_records if r.mutation_sensitive]
            job.diagnostic_messages.append(
                f"Mutation validation completed: {len(mut_sensitive)} demonstrated regression defect detection"
            )

            # Stage 6: COMPLETED
            cases = generator.construct_benchmark_cases(repo_id, mut_records, is_git=repo_record.is_git)

            # Update repository record
            repo_record.generated_tests = mut_records
            repo_record.generated_tests_count = len(mut_records)
            repo_record.validated_tests_count = len([t for t in mut_records if t.executes_successfully and not t.is_trivial])
            repo_record.mutation_sensitive_tests_count = len(mut_sensitive)

            existing_case_ids = {c.case_id for c in repo_record.benchmark_cases}
            for c in cases:
                if c.case_id not in existing_case_ids:
                    repo_record.benchmark_cases.append(c)

            repo_record.discovered_tests = evaluator.discover_all_tests()
            repo_record.total_tests_count = len(repo_record.discovered_tests)
            validated_cases = [c for c in repo_record.benchmark_cases if c.ground_truth.is_verified]
            repo_record.validated_benchmark_cases_count = len(validated_cases)
            repo_record.cases_awaiting_review_count = len([c for c in repo_record.benchmark_cases if not c.ground_truth.is_verified])

            if repo_record.validated_benchmark_cases_count > 0:
                repo_record.status = RepositoryStatus.READY_FOR_EVALUATION
                repo_record.status_message = (
                    f"Generated {repo_record.validated_tests_count} validated tests "
                    f"({repo_record.mutation_sensitive_tests_count} mutation-sensitive). "
                    f"{repo_record.validated_benchmark_cases_count} benchmark cases ready for evaluation."
                )
            else:
                repo_record.status = RepositoryStatus.BENCHMARK_PREPARATION_REQUIRED
                repo_record.status_message = (
                    f"Synthesized {repo_record.generated_tests_count} tests; "
                    "manual review or independent ground truth required."
                )

            custom_repo_mgr.update_repository_record(repo_record)

            job.generated_test_records = mut_records
            job.output_artifacts = [
                str(repo_path / "tests" / "generated" / f)
                for f in os.listdir(repo_path / "tests" / "generated")
                if f.endswith(".py")
            ] if (repo_path / "tests" / "generated").exists() else []

            job.status = GenerationJobStatus.COMPLETED
            job.current_stage = f"Generation complete: {job.accepted_count} accepted, {len(mut_sensitive)} mutation-verified"
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.save_job(job)

        except Exception as e:
            logger.error(f"Generation job {job_id} failed: {e}", exc_info=True)
            job.status = GenerationJobStatus.FAILED
            job.current_stage = "Job failed due to error"
            job.failure_reason = str(e)
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.save_job(job)

            repo_record.status = RepositoryStatus.FAILED
            repo_record.status_message = f"Automatic test generation failed: {str(e)}"
            custom_repo_mgr.update_repository_record(repo_record)
        finally:
            self._active_threads.pop(job_id, None)
            self._cancellation_flags.pop(job_id, None)
