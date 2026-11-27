"""Synthetic worker/oneshot permission regressions. No agy or auth access."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

import client as agy_client
from process import read_media_rules, write_media_read_permissions


MODEL = "gemini-3.8-flash-high"


def rules_for(*paths):
    return frozenset(
        f"{tool}({path})" for path in paths for tool in ("read_file", "view_file")
    )


@pytest.fixture
def worker(tmp_path, monkeypatch):
    # Bypass construction: no command discovery, orphan reaping, home setup,
    # keyring access, timers, token sync or real subprocesses in this fixture.
    cl = agy_client.AntigravityClient.__new__(agy_client.AntigravityClient)
    # PR #46's cancel fence lives on the instance surface; bypassing __init__
    # must still provide it.
    cl._cancelled = False
    cl.is_closed = False
    cl._lock = threading.RLock()
    cl._worker_lock = threading.Lock()
    cl._isolated_gemini_dir = tmp_path
    cl._isolated_home = tmp_path
    cl._cwd = str(tmp_path)
    cl._temp_dir = None
    cl._command = "synthetic-agy"
    cl._args = []
    cl._materialized_media = set()
    proc = MagicMock(name="existing_worker")
    proc.poll.return_value = None
    cl._worker_proc = proc
    cl._worker_model = MODEL
    cl._worker_effort = None
    cl._worker_history = [{"role": "user", "content": "previous turn"}]
    cl._conversation_history = list(cl._worker_history)
    cl._worker_conversation_id = "synthetic-conversation"
    cl._worker_media_rules = frozenset()
    cl._worker_usage_baseline = {"input_tokens": 123}
    cl._active_processes = {proc}
    cl._worker_last_activity = {proc: 1.0}
    cl._worker_cpu_at_activity = {proc: 0.0}
    cl._delivery_watch = None
    cl._terminate_process = MagicMock()
    cl._cancel_idle_timer_locked = MagicMock()
    cl._arm_idle_timer_locked = MagicMock()
    cl._child_env = lambda: {}
    monkeypatch.setattr(agy_client, "is_authenticated", lambda: True)
    monkeypatch.setattr(cl, "_resolve_model_and_effort", lambda *args: (MODEL, None))
    monkeypatch.setattr(agy_client, "sync_isolated_token", MagicMock())
    monkeypatch.setattr(agy_client, "start_stderr_drainer", MagicMock())
    monkeypatch.setattr(agy_client, "_worker_cpu_seconds", lambda proc: None)
    monkeypatch.setattr(agy_client, "dump_prompt_debug", MagicMock())
    spawned = MagicMock(name="spawned_process")
    spawned.poll.return_value = None

    def spawn(*args, **kwargs):
        spawned.media_rules_at_spawn = read_media_rules(tmp_path)
        # agy announces the resumed conversation at start-up, before any input.
        conv = args[0][args[0].index("--conversation") + 1] if "--conversation" in args[0] else "new"
        spawned.stdout.readline.side_effect = [
            json.dumps({"event": "init", "conversation_id": conv}) + "\n",
            "",
        ]
        return spawned

    monkeypatch.setattr(agy_client.subprocess, "Popen", MagicMock(side_effect=spawn))
    monkeypatch.setattr(agy_client, "AntigravityStream", MagicMock())
    return cl, proc, spawned


def image_turn(cl, tmp_path, name="new.png"):
    path = tmp_path / name
    path.write_bytes(b"synthetic image")
    return str(path), cl._conversation_history + [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": str(path)}}],
    }]


def test_busy_worker_image_oneshot_does_not_terminate_or_stamp_worker(worker, tmp_path):
    cl, proc, spawned = worker
    path, messages = image_turn(cl, tmp_path)
    baseline = cl._worker_usage_baseline
    history = list(cl._worker_history)
    cl._worker_lock.acquire()  # Represents another request's active stream.
    try:
        cl._create_chat_completion(model=MODEL, messages=messages, stream=True)
        cl._terminate_process.assert_not_called()
        assert cl._worker_proc is proc
        assert cl._worker_lock.locked()
        assert cl._worker_history == history
        assert cl._worker_conversation_id == "synthetic-conversation"
        assert cl._worker_usage_baseline is baseline
        assert cl._worker_media_rules == frozenset()
        assert read_media_rules(tmp_path) == rules_for(path)
        assert spawned.media_rules_at_spawn == rules_for(path)
        assert agy_client.AntigravityStream.call_args.kwargs["proc"] is spawned
        assert agy_client.AntigravityStream.call_args.kwargs["is_worker"] is False
        assert "worker_lock" not in agy_client.AntigravityStream.call_args.kwargs
        assert "--conversation" not in agy_client.subprocess.Popen.call_args.args[0]
        # Permission references must not become cleanup-owned user originals.
        assert path not in cl._materialized_media
        assert (tmp_path / "new.png").is_file()
    finally:
        cl._worker_lock.release()

    # Disk permissions from the oneshot do not mean the old worker loaded them.
    cl._create_chat_completion(model=MODEL, messages=messages, stream=True)
    try:
        cl._terminate_process.assert_called_once_with(proc)
        assert cl._worker_proc is spawned
        assert cl._worker_media_rules == rules_for(path)
        assert "synthetic-conversation" in agy_client.subprocess.Popen.call_args.args[0]
    finally:
        cl._worker_lock.release()


def test_already_allowed_media_reuses_worker_despite_other_disk_rules(worker, tmp_path):
    cl, proc, _ = worker
    path, messages = image_turn(cl, tmp_path)
    cl._worker_media_rules = write_media_read_permissions(tmp_path, {path})
    # A concurrent oneshot can have added rules the worker never loaded. Those
    # unrelated rules must not force a restart for an already-allowed image.
    write_media_read_permissions(tmp_path, {str(tmp_path / "oneshot-only.png")})
    snapshot = cl._worker_media_rules
    cl._create_chat_completion(model=MODEL, messages=messages, stream=True)
    try:
        cl._terminate_process.assert_not_called()
        agy_client.subprocess.Popen.assert_not_called()
        assert cl._worker_proc is proc
        assert cl._worker_media_rules is snapshot
    finally:
        cl._worker_lock.release()


@pytest.mark.parametrize("partial", [False, True])
def test_new_media_recycles_idle_worker_and_resumes_conversation(worker, tmp_path, partial):
    cl, proc, spawned = worker
    path, messages = image_turn(cl, tmp_path)
    if partial:
        # Both tool rules are required, not merely one read_file grant.
        cl._worker_media_rules = frozenset({f"read_file({path})"})
    history = list(cl._conversation_history)
    original_terminate = cl._terminate_worker_locked

    def terminate(*, keep_conversation=False):
        assert cl._worker_lock.locked()
        return original_terminate(keep_conversation=keep_conversation)

    cl._terminate_worker_locked = terminate
    old_baseline = cl._worker_usage_baseline
    cl._create_chat_completion(model=MODEL, messages=messages, stream=True)
    try:
        cl._terminate_process.assert_called_once_with(proc)
        assert cl._worker_proc is spawned
        assert cl._worker_conversation_id == "synthetic-conversation"
        assert cl._conversation_history == history
        assert cl._worker_history == history
        # The resumed conversation's counters continue from the session
        # total, so the respawned worker starts from a copy of the snapshot.
        assert cl._worker_usage_baseline == {"input_tokens": 123}
        assert cl._worker_usage_baseline is not old_baseline
        assert cl._worker_media_rules == rules_for(path)
        args = agy_client.subprocess.Popen.call_args.args[0]
        assert args[args.index("--conversation") + 1] == "synthetic-conversation"
        assert "--dangerously-skip-permissions" not in args
        settings = json.loads((tmp_path / "settings.json").read_text())
        assert set(settings["permissions"]["allow"]) == rules_for(path)
        assert path not in cl._materialized_media
    finally:
        cl._worker_lock.release()
    cl._terminate_worker_locked = original_terminate
    cl.close()
    assert (tmp_path / "new.png").read_bytes() == b"synthetic image"


def test_concurrent_permission_merges_are_serialized(worker, tmp_path, monkeypatch):
    cl, proc, _ = worker
    first_entered = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    paths = [str(tmp_path / name) for name in ("first.png", "second.png")]

    def write(directory, requested):
        if paths[0] in requested:
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        return write_media_read_permissions(directory, requested)

    def second():
        second_started.set()
        cl._ensure_worker_media_permissions({paths[1]}, worker_turn=False)

    monkeypatch.setattr(agy_client, "write_media_read_permissions", write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cl._ensure_worker_media_permissions, {paths[0]}, worker_turn=False)
        try:
            assert first_entered.wait(5)
            other = pool.submit(second)
            assert second_started.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release_first.set()
        first.result(timeout=5)
        other.result(timeout=5)
    assert second_entered.is_set()
    assert read_media_rules(tmp_path) == rules_for(*paths)
    assert cl._worker_proc is proc
    assert cl._worker_media_rules == frozenset()
    cl._terminate_process.assert_not_called()


@pytest.mark.parametrize("worker_turn", [False, True])
@pytest.mark.parametrize("snapshot_current", [False, True])
def test_disk_granted_media_skips_write_but_checks_worker_snapshot(
    worker, tmp_path, monkeypatch, worker_turn, snapshot_current
):
    cl, proc, _ = worker
    path, _ = image_turn(cl, tmp_path)
    disk_rules = write_media_read_permissions(tmp_path, {path})
    if snapshot_current:
        cl._worker_media_rules = disk_rules
    snapshot = cl._worker_media_rules
    write = MagicMock(wraps=write_media_read_permissions)
    monkeypatch.setattr(agy_client, "write_media_read_permissions", write)
    # A busy oneshot does not own this lock, even though it is held.
    cl._worker_lock.acquire()
    try:
        cl._ensure_worker_media_permissions({path}, worker_turn=worker_turn)
        assert cl._worker_lock.locked()
        write.assert_not_called()
        assert read_media_rules(tmp_path) == disk_rules
        if worker_turn and not snapshot_current:
            cl._terminate_process.assert_called_once_with(proc)
            assert cl._worker_proc is None
            assert cl._worker_conversation_id == "synthetic-conversation"
        else:
            cl._terminate_process.assert_not_called()
            assert cl._worker_proc is proc
            assert cl._worker_media_rules is snapshot
    finally:
        cl._worker_lock.release()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("busy", [False, True])
def test_settings_write_failure_preserves_worker_and_turn_lock_ownership(
    worker, tmp_path, monkeypatch, stream, busy
):
    cl, proc, _ = worker
    path, messages = image_turn(cl, tmp_path)
    history = cl._worker_history
    conversation_history = cl._conversation_history
    baseline = cl._worker_usage_baseline
    snapshot = cl._worker_media_rules
    failure = OSError("synthetic settings write failure")
    write = MagicMock(side_effect=failure)
    monkeypatch.setattr(agy_client, "write_media_read_permissions", write)
    if busy:
        cl._worker_lock.acquire()
    try:
        with pytest.raises(OSError, match="synthetic settings write failure") as exc:
            cl._create_chat_completion(model=MODEL, messages=messages, stream=stream)
        assert exc.value is failure
        write.assert_called_once_with(tmp_path, {path})
        cl._terminate_process.assert_not_called()
        agy_client.subprocess.Popen.assert_not_called()
        agy_client.AntigravityStream.assert_not_called()
        proc.stdin.write.assert_not_called()
        assert cl._worker_proc is proc
        assert cl._worker_history is history
        assert cl._conversation_history is conversation_history
        assert history == [{"role": "user", "content": "previous turn"}]
        assert conversation_history == history
        assert cl._worker_conversation_id == "synthetic-conversation"
        assert cl._worker_usage_baseline is baseline
        assert cl._worker_media_rules is snapshot
        assert cl._worker_lock.locked() is busy
        if not busy:
            assert cl._worker_lock.acquire(blocking=False)
            cl._worker_lock.release()
    finally:
        if busy:
            cl._worker_lock.release()


def test_empty_media_does_not_write_or_recycle(worker, monkeypatch):
    cl, proc, _ = worker
    write = MagicMock()
    monkeypatch.setattr(agy_client, "write_media_read_permissions", write)
    cl._ensure_worker_media_permissions(set(), worker_turn=False)
    cl._worker_lock.acquire()
    try:
        cl._ensure_worker_media_permissions(set(), worker_turn=True)
    finally:
        cl._worker_lock.release()
    write.assert_not_called()
    cl._terminate_process.assert_not_called()
    assert cl._worker_proc is proc
