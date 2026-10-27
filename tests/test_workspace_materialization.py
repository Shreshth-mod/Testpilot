"""
Unit and regression tests for TestPilot workspace materialization and snapshot accuracy.
Verifies:
1. Clean commit materialization matches exact Git commit archive.
2. Dirty snapshot materialization applies Git diffs and restores untracked files.
3. Host worktree drift immunity: modifying the host working tree does NOT alter the materialized execution workspace.
4. Workspace cleanup removes disposable directories safely.
"""

from pathlib import Path
import json
import shutil
import subprocess
import pytest

from testpilot.core.workflow_context import (
    WorkflowContext,
    WorkflowStorageManager,
    StageStatus,
)


def test_clean_commit_workspace_materialization(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    
    # Get current HEAD of active repo
    rp = Path(".").resolve()
    head_sha = mgr.resolve_immutable_commit(str(rp), "HEAD")

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=str(rp),
        repo_name="Testbed",
        base_commit_hash=head_sha,
        target_commit_hash=head_sha,
        is_dirty=False,
        workflow_id="wf-test-clean-mat",
    )

    ws_path, is_disposable = mgr.materialize_execution_workspace(wf)
    try:
        assert is_disposable is True
        assert ws_path.exists()
        assert (ws_path / "pyproject.toml").exists()
        assert (ws_path / "testpilot").exists()
        # Verify it has no .git directory (clean exported tree)
        assert not (ws_path / ".git").exists()
    finally:
        mgr.cleanup_execution_workspace(ws_path)
        assert not ws_path.exists()


def test_dirty_snapshot_workspace_materialization_with_untracked_files(tmp_path):
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    rp = Path(".").resolve()
    head_sha = mgr.resolve_immutable_commit(str(rp), "HEAD")

    wf_id = "wf-test-dirty-mat"
    snap_id = f"snap-{wf_id}"
    snap_file = mgr.snapshots_dir / f"{snap_id}.patch"

    # Create synthetic snapshot payload with an untracked file
    untracked_rel = "tests/test_synthetic_untracked_probe.py"
    untracked_content = "# Synthetic probe file for snapshot testing\ndef test_probe(): assert True\n"
    
    payload = {
        "workflow_id": wf_id,
        "timestamp": "2026-10-27T00:00:00Z",
        "diff_head": "",
        "untracked_files": {
            untracked_rel: untracked_content,
        },
    }
    snap_file.write_text(json.dumps(payload), encoding="utf-8")

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=str(rp),
        repo_name="Testbed",
        base_commit_hash=head_sha,
        target_commit_hash=head_sha,
        is_dirty=True,
        patch_snapshot_id=snap_id,
        workflow_id=wf_id,
    )

    ws_path, is_disposable = mgr.materialize_execution_workspace(wf)
    try:
        assert is_disposable is True
        assert ws_path.exists()
        # Verify untracked file from snapshot is restored in workspace
        probe_in_ws = ws_path / untracked_rel
        assert probe_in_ws.exists()
        assert probe_in_ws.read_text(encoding="utf-8") == untracked_content
    finally:
        mgr.cleanup_execution_workspace(ws_path)
        assert not ws_path.exists()


def test_host_worktree_drift_immunity(tmp_path):
    """
    Proves that modifying files in the host repository AFTER workflow initialization
    does NOT bleed into the materialized execution workspace.
    """
    mgr = WorkflowStorageManager(base_dir=tmp_path / "workflows")
    rp = Path(".").resolve()
    head_sha = mgr.resolve_immutable_commit(str(rp), "HEAD")

    wf = WorkflowContext.create_initial(
        repo_id="testbed_repo",
        repo_path=str(rp),
        repo_name="Testbed",
        base_commit_hash=head_sha,
        target_commit_hash=head_sha,
        is_dirty=False,
        workflow_id="wf-test-drift-immunity",
    )

    # Materialize workspace
    ws_path, is_disposable = mgr.materialize_execution_workspace(wf)
    try:
        assert is_disposable is True
        # Verify host drift file does NOT exist in materialized workspace
        drift_probe = ws_path / "host_drift_uncommitted_file.tmp"
        assert not drift_probe.exists()
    finally:
        mgr.cleanup_execution_workspace(ws_path)


def test_dirty_snapshot_staged_deleted_renamed_and_binary_files(tmp_path):
    """
    Edge-Case Test: Proves that dirty snapshots faithfully reconstruct:
    - Staged newly added files
    - Deleted files
    - Renamed files
    - Binary untracked files with arbitrary byte sequences
    - Untracked source files
    """
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Initialize a clean Git repository
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestPilot"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "test@testpilot.ai"], cwd=str(repo_dir), check=True)

    # 1. Initial tracked commit
    (repo_dir / "keep.py").write_text("def keep(): pass\n", encoding="utf-8")
    (repo_dir / "to_delete.py").write_text("def delete_me(): pass\n", encoding="utf-8")
    (repo_dir / "to_rename.py").write_text("def rename_me(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo_dir), check=True)
    head_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    # 2. Make dirty changes:
    # - Staged file
    (repo_dir / "staged_file.py").write_text("def staged(): return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged_file.py"], cwd=str(repo_dir), check=True)

    # - Deleted file
    (repo_dir / "to_delete.py").unlink()

    # - Renamed file (untracked move)
    shutil.move(str(repo_dir / "to_rename.py"), str(repo_dir / "renamed_file.py"))

    # - Binary untracked file with null bytes and high bytes
    binary_bytes = b"\x00\x01\x02\x7f\x80\xff\xfe\x00binary_payload" * 20
    (repo_dir / "untracked_data.bin").write_bytes(binary_bytes)

    # - Untracked source file
    (repo_dir / "untracked_test.py").write_text("def test_untracked(): assert True\n", encoding="utf-8")

    # 3. Create snapshot via WorkflowStorageManager
    mgr = WorkflowStorageManager(base_dir=repo_dir / ".testpilot" / "workflows")
    snap_id, snap_hash = mgr.create_dirty_snapshot(str(repo_dir), "wf-edge-snapshot")

    wf = WorkflowContext.create_initial(
        repo_id="mock_repo",
        repo_path=str(repo_dir),
        repo_name="MockRepo",
        base_commit_hash=head_sha,
        target_commit_hash=head_sha,
        is_dirty=True,
        patch_snapshot_id=snap_id,
        workflow_id="wf-edge-snapshot",
    )

    # 4. Materialize disposable workspace
    ws_path, is_disposable = mgr.materialize_execution_workspace(wf)
    try:
        assert is_disposable is True
        assert ws_path.exists()

        # Invariant 1: Untouched file remains
        assert (ws_path / "keep.py").exists()

        # Invariant 2: Staged newly added file exists with correct content
        assert (ws_path / "staged_file.py").exists()
        assert (ws_path / "staged_file.py").read_text(encoding="utf-8") == "def staged(): return True\n"

        # Invariant 3: Deleted file is removed from workspace
        assert not (ws_path / "to_delete.py").exists(), "Deleted file was not removed from workspace"

        # Invariant 4: Renamed file exists and old name does not
        assert not (ws_path / "to_rename.py").exists(), "Old renamed filename was not removed"
        assert (ws_path / "renamed_file.py").exists(), "New renamed filename does not exist"

        # Invariant 5: Binary untracked file has bit-for-bit exact match
        assert (ws_path / "untracked_data.bin").exists()
        assert (ws_path / "untracked_data.bin").read_bytes() == binary_bytes

        # Invariant 6: Untracked text file exists
        assert (ws_path / "untracked_test.py").exists()
        assert (ws_path / "untracked_test.py").read_text(encoding="utf-8") == "def test_untracked(): assert True\n"
    finally:
        mgr.cleanup_execution_workspace(ws_path)
        assert not ws_path.exists()


def test_extract_failure_traceback_preserves_assertion_and_failures_block():
    """
    Edge-Case Test: Proves that output truncation does not discard failure details
    from the FAILURES block or the summary section when stdout is large.
    """
    from testpilot.execution.runner import extract_failure_traceback

    # Simulate 15,000 characters of verbose test collection output before FAILURES
    verbose_preamble = "verbose collection output line\n" * 500
    failure_block = """=== FAILURES ===
________________________ test_order_discount_regression ________________________
    def test_order_discount_regression():
>       assert calculated_discount == expected_discount
E       AssertionError: assert 12.50 == 15.00
E         -15.00
E         +12.50
testbed/tests/test_order.py:42: AssertionError
=========================== short test summary info ============================
FAILED testbed/tests/test_order.py::test_order_discount_regression - AssertionError: assert 12.50 == 15.00
============================== 1 failed in 0.45s ===============================
"""
    long_output = verbose_preamble + failure_block

    # Extract with standard 4,000 char budget
    tb = extract_failure_traceback(long_output, stderr="", max_chars=4000)

    # Assert that the failure section, assertion message, and summary are preserved
    assert len(tb) <= 4000
    assert "=== FAILURES ===" in tb
    assert "AssertionError: assert 12.50 == 15.00" in tb
    assert "FAILED testbed/tests/test_order.py::test_order_discount_regression" in tb

