"""Hermes Stop regressions using only in-memory subprocesses and fake host seams.

Run independently of Hermes with ``python tests/test_cancel.py``. Hermes marks
the request slot poisoned before dispatching the provider's cancel() hook, so
continuation uses a fresh client, never the cancelled facade.
"""

import asyncio
from collections import deque
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client as client_module
from client import AntigravityClient


MESSAGES = [{"role": "user", "content": "synthetic stopped request"}]


def _lines(text="synthetic answer"):
    return [json.dumps(event) + "\n" for event in (
        {"event": "init", "conversation_id": "synthetic-conversation"},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {"event": "result", "result": {"status": "SUCCESS", "response": text}},
    )]


class _Input:
    def __init__(self):
        self.writes = []
        self.on_write = None

    def write(self, value):
        if self.on_write:
            self.on_write()
        self.writes.append(value)

    def flush(self):
        pass

    def close(self):
        pass


class _Process:
    """No PID, descriptors, child process, or host signal of any kind."""

    pid = None

    def __init__(self, lines=(), *, blocked=False):
        self.stdin = _Input()
        self.stdout = self
        self.stderr = io.StringIO()
        self.lines = deque(lines)
        self.blocked = blocked
        self.entered = threading.Event()
        self.reaped = threading.Event()
        self.alive = True
        self.terminate_gate = None

    def readline(self):
        self.entered.set()
        if self.blocked:
            if not self.reaped.wait(2):
                raise AssertionError("fake read was never stopped")
            return ""
        # Deliberately retain buffered SUCCESS after death. Killing alone is
        # insufficient: the consumer must reject the cancelled stream too.
        return self.lines.popleft() if self.lines else ""

    def poll(self):
        return None if self.alive else 0


class CancelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="agy-cancel-fixture-")
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        self.processes = []
        self.spawn_calls = []
        self.clients = []
        self.pending = deque()
        seams = {
            "setup_isolated_home": lambda cwd: (self.workspace / "home", None),
            "reap_orphaned_plugin_workers": lambda: None,
            "is_authenticated": lambda: True,
            "resolve_agy_command": lambda: "synthetic-agy",
            "resolve_model_and_effort": lambda model, effort: (model or "synthetic-model", effort),
            "sync_isolated_token": lambda home: None,
            "start_stderr_drainer": lambda proc: None,
            "_own_process_group": lambda: {},
            "build_child_env": lambda home: {},
            "read_media_rules": lambda home: frozenset(),
            "_worker_cpu_seconds": lambda proc: None,
            "dump_prompt_debug": lambda **kwargs: None,
        }
        for name, replacement in seams.items():
            patcher = patch.object(client_module, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, replacement in (
            ("_terminate_process", staticmethod(self._terminate)),
            ("_arm_idle_timer_locked", lambda self: None),
            ("_purge_isolated_conversations", lambda self: None),
        ):
            patcher = patch.object(AntigravityClient, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(client_module.subprocess, "Popen", self._spawn)
        patcher.start()
        self.addCleanup(patcher.stop)
        # LIFO: release blocked fakes before closing clients or undoing seams.
        self.addCleanup(self._cleanup)

    def _spawn(self, args, **kwargs):
        self.spawn_calls.append((args, kwargs))
        if not self.pending:
            raise AssertionError("unexpected process spawn/replay")
        proc = self.pending.popleft()
        self.processes.append(proc)
        return proc

    @staticmethod
    def _terminate(proc):
        if proc.terminate_gate:
            proc.terminate_gate.wait(2)
        proc.alive = False
        proc.reaped.set()

    def _cleanup(self):
        for proc in self.processes:
            if proc.terminate_gate:
                proc.terminate_gate.set()
            self._terminate(proc)
        for client in self.clients:
            client.close()

    def _client(self):
        client = AntigravityClient(cwd=str(self.workspace))
        self.clients.append(client)
        return client

    def _create(self, client, **kwargs):
        return client.chat.completions.create(
            model="synthetic-model", messages=kwargs.pop("messages", MESSAGES),
            stream=kwargs.pop("stream", True), **kwargs,
        )

    @staticmethod
    def _stop(client):
        # Same non-HTTP dispatch used by ClientLifecycleMixin after poisoning
        # the request slot. A provider without the hook cannot stop its child.
        if callable(getattr(type(client), "cancel", None)):
            client.cancel()

    def test_stop_unblocks_owned_read(self):
        client = self._client()
        proc = _Process(blocked=True)
        self.pending.append(proc)
        stream = self._create(client)
        finished = threading.Event()
        errors = []

        def consume():
            try:
                list(stream)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        self.assertTrue(proc.entered.wait(1))
        try:
            self._stop(client)
            self.assertTrue(finished.wait(0.5), "Stop left its request running")
            self.assertTrue(proc.reaped.is_set())
            self.assertEqual(len(errors), 1)
            self.assertRegex(str(errors[0]), "cancelled")
            self.assertFalse(client._worker_lock.locked())
            self.assertEqual(client._conversation_history, [])
        finally:
            self._terminate(proc)
            reader.join(1)
            stream.close()

    def test_ordinary_success_still_reuses_the_worker(self):
        client = self._client()
        proc = _Process(_lines() + _lines("continued answer"))
        self.pending.append(proc)
        list(self._create(client))
        transcript = MESSAGES + [{"role": "user", "content": "synthetic continuation"}]
        list(self._create(client, messages=transcript))
        self.assertTrue(proc.alive)
        self.assertEqual(len(self.spawn_calls), 1)
        self.assertEqual(len(proc.stdin.writes), 2)
        self.assertEqual(client._worker_history, transcript)
        self.assertFalse(client._worker_lock.locked())

    def test_ordinary_broken_pipe_still_retries_once(self):
        client = self._client()
        failed = _Process()
        replacement = _Process(_lines())
        self.pending.extend((failed, replacement))

        def break_pipe():
            raise BrokenPipeError("synthetic ordinary write failure")

        failed.stdin.on_write = break_pipe
        list(self._create(client))
        self.assertEqual(len(self.spawn_calls), 2)
        self.assertTrue(failed.reaped.is_set())
        self.assertTrue(replacement.alive)
        self.assertEqual(len(replacement.stdin.writes), 1)

    def test_stop_rejects_buffered_success_before_first_read(self):
        client = self._client()
        self.pending.append(_Process(_lines()))
        stream = self._create(client)
        self._stop(client)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            list(stream)
        self.assertEqual(client._worker_history, [])
        self.assertIsNone(client._worker_conversation_id)
        self.assertFalse(client._worker_lock.locked())

    def test_stop_after_partial_output_rejects_success(self):
        client = self._client()
        self.pending.append(_Process(_lines()))
        stream = self._create(client)
        self.assertEqual(next(stream).choices[0].delta.content, "synthetic answer")
        self._stop(client)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            list(stream)
        self.assertEqual(client._conversation_history, [])
        # Finalizing the paused generator must not restore stopped history.
        stream._generator.close()
        self.assertEqual(client._conversation_history, [])

    def test_stop_during_worker_write_never_replays(self):
        client = self._client()
        proc = _Process(_lines())
        self.pending.extend((proc, _Process(_lines("unexpected replay"))))

        def stop_then_break_pipe():
            self._stop(client)
            raise BrokenPipeError("synthetic Stop during write")

        proc.stdin.on_write = stop_then_break_pipe
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self._create(client)
        self.assertEqual(len(self.spawn_calls), 1)
        self.assertTrue(proc.reaped.is_set())
        self.assertFalse(client._worker_lock.locked())

    def test_stop_after_finish_chunk_does_not_commit_late_history(self):
        client = self._client()
        self.pending.append(_Process(_lines()))
        stream = self._create(client)
        next(stream)
        self.assertEqual(next(stream).choices[0].finish_reason, "stop")
        self._stop(client)
        stream._generator.close()
        self.assertEqual(client._conversation_history, [])
        self.assertIsNone(client._worker_conversation_id)

    def test_stop_after_replacement_spawn_never_writes_retry_prompt(self):
        client = self._client()
        failed = _Process()
        replacement = _Process(_lines())
        self.pending.extend((failed, replacement))

        def break_pipe():
            raise BrokenPipeError("synthetic ordinary write failure")

        def stop_on_retry_dump(**kwargs):
            if kwargs["branch"] == "retry":
                self._stop(client)

        failed.stdin.on_write = break_pipe
        with patch.object(client_module, "dump_prompt_debug", stop_on_retry_dump):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self._create(client)
        self.assertEqual(len(self.spawn_calls), 2)
        self.assertEqual(replacement.stdin.writes, [])
        self.assertTrue(replacement.reaped.is_set())

    def test_stop_during_contention_retry_cannot_return_old_completion(self):
        client = self._client()
        failed = _Process(_lines())
        replacement = _Process(_lines())
        self.pending.extend((failed, replacement))
        # Pure synthetic retry classification, no stderr or credentials read.
        empty = client_module.SimpleNamespace(choices=[])
        real_spawn = client._get_or_spawn_worker
        calls = []

        def spawn_then_stop(*args, **kwargs):
            proc = real_spawn(*args, **kwargs)
            calls.append(proc)
            if len(calls) == 2:
                self._stop(client)
            return proc

        with patch.object(client_module, "collect_stream_completion", return_value=empty), \
                patch.object(client_module, "_empty_no_result", return_value=True), \
                patch.object(client_module, "_stderr_shows_auth_contention", return_value=True), \
                patch.object(client, "_get_or_spawn_worker", spawn_then_stop):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self._create(client, stream=False)
        self.assertEqual(len(self.spawn_calls), 2)
        self.assertEqual(replacement.stdin.writes, [])

    def test_oneshot_is_owned_before_stdin_write(self):
        client = self._client()
        client._worker_lock.acquire()
        self.addCleanup(client._worker_lock.release)
        proc = _Process(_lines())
        self.pending.append(proc)
        owned_at_write = []

        def stop_during_write():
            owned_at_write.append(proc in client._active_processes)
            self._stop(client)
            raise BrokenPipeError("synthetic oneshot Stop")

        proc.stdin.on_write = stop_during_write
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self._create(client)
        self.assertEqual(owned_at_write, [True])
        self.assertTrue(proc.reaped.is_set())
        self.assertEqual(len(self.spawn_calls), 1)

    def test_next_hermes_client_starts_clean_without_resume_or_replay(self):
        old = self._client()
        stopped_proc = _Process(_lines())
        next_proc = _Process(_lines("next answer"))
        self.pending.extend((stopped_proc, next_proc))
        old_stream = self._create(old)
        next(old_stream)
        self._stop(old)
        self.assertTrue(stopped_proc.reaped.is_set())
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self._create(old)
        self.assertFalse(old.is_closed, "owner close still owns workspace cleanup")
        fresh = self._client()
        transcript = MESSAGES + [{"role": "user", "content": "synthetic next request"}]
        fresh_stream = self._create(fresh, messages=transcript)
        old_stream.close()
        self.assertTrue(next_proc.alive, "stale cleanup killed a fresh client's worker")
        chunks = list(fresh_stream)
        self.assertTrue(any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks))
        self.assertEqual(len(self.spawn_calls), 2)
        self.assertNotIn("--conversation", self.spawn_calls[1][0])
        self.assertEqual(len(next_proc.stdin.writes), 1)
        payload = json.loads(next_proc.stdin.writes[0])["message"]["content"]
        self.assertEqual(payload, client_module._format_messages_as_prompt(
            transcript, model="synthetic-model", media_sink=fresh._materialized_media,
        ))
        self.assertEqual(fresh._worker_history, transcript)

    def test_nonstream_stop_during_collection_cannot_return_success(self):
        client = self._client()
        proc = _Process(_lines())
        self.pending.append(proc)
        real_collect = client_module.collect_stream_completion

        def stop_after_collect(stream):
            completion = real_collect(stream)
            self._stop(client)
            return completion

        with patch.object(client_module, "collect_stream_completion", stop_after_collect):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self._create(client, stream=False)
        self.assertEqual(len(self.spawn_calls), 1)
        self.assertEqual(client._conversation_history, [])

    def test_cancel_in_async_loop_does_not_wait_for_process_cleanup(self):
        client = self._client()
        proc = _Process(_lines())
        self.pending.append(proc)
        stream = self._create(client)
        proc.terminate_gate = threading.Event()
        release = threading.Timer(0.3, proc.terminate_gate.set)
        release.daemon = True
        release.start()
        try:
            async def stop_in_loop():
                start = time.monotonic()
                self._stop(client)
                self.assertLess(time.monotonic() - start, 0.15)
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    await stream.__anext__()

            asyncio.run(stop_in_loop())
        finally:
            proc.terminate_gate.set()
            release.cancel()
            self.assertTrue(proc.reaped.wait(1))
            stream.close()

    def test_stop_unblocks_pending_async_read(self):
        client = self._client()
        proc = _Process(blocked=True)
        self.pending.append(proc)
        stream = self._create(client)

        async def consume():
            task = asyncio.create_task(stream.__anext__())
            try:
                deadline = time.monotonic() + 1
                while not proc.entered.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertTrue(proc.entered.is_set())
                self._stop(client)
                deadline = time.monotonic() + 0.5
                while not task.done() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertTrue(task.done(), "Stop left the async read running")
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    await task
            finally:
                self._terminate(proc)
                stream.close()
                if not task.done():
                    task.cancel()

        asyncio.run(consume())
        self.assertTrue(proc.reaped.is_set())

    def test_owner_close_waits_for_cancelled_oneshot_before_workspace_cleanup(self):
        client = self._client()
        client._worker_lock.acquire()
        self.addCleanup(client._worker_lock.release)
        proc = _Process(_lines())
        self.pending.append(proc)
        stream = self._create(client)
        proc.terminate_gate = threading.Event()
        termination_started = threading.Event()
        cleanup_started = threading.Event()
        real_terminate = self._terminate

        def gated_terminate(child):
            termination_started.set()
            real_terminate(child)

        def remove_workspace():
            self.assertFalse(proc.alive, "workspace cleanup raced a live oneshot")
            cleanup_started.set()

        async def stop_in_loop():
            self._stop(client)

        with patch.object(client, "_terminate_process", gated_terminate), \
                patch.object(client, "_remove_temp_dir", remove_workspace):
            asyncio.run(stop_in_loop())
            self.assertTrue(termination_started.wait(1))
            owner = threading.Thread(target=client.close, daemon=True)
            owner.start()
            try:
                self.assertFalse(cleanup_started.wait(0.05))
            finally:
                proc.terminate_gate.set()
                owner.join(1)
            self.assertFalse(owner.is_alive())
            self.assertTrue(cleanup_started.is_set())
        stream.close()

    def test_stop_rejects_async_chunk_waiting_for_delivery(self):
        client = self._client()
        self.pending.append(_Process(_lines()))
        stream = self._create(client)

        def queued_chunk_then_stop():
            chunk = stream._make_chunk(content="synthetic queued chunk")
            self._stop(client)
            return False, chunk

        async def consume():
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                await stream.__anext__()

        with patch.object(stream, "_pull_chunk", queued_chunk_then_stop):
            asyncio.run(consume())
        stream.close()

    def test_owner_close_waits_when_stream_cleanup_wins_cancel_race(self):
        client = self._client()
        client._worker_lock.acquire()
        self.addCleanup(client._worker_lock.release)
        proc = _Process(_lines())
        self.pending.append(proc)
        stream = self._create(client)
        proc.terminate_gate = threading.Event()
        cancel_gate = threading.Event()
        cancel_done = threading.Event()
        termination_started = threading.Event()
        cleanup_started = threading.Event()
        self.addCleanup(cancel_gate.set)
        self.addCleanup(proc.terminate_gate.set)
        real_cancel_cleanup = getattr(client, "_cancel_owned_requests", lambda: None)

        def delayed_cancel_cleanup():
            cancel_gate.wait(2)
            try:
                real_cancel_cleanup()
            finally:
                cancel_done.set()

        def gated_terminate(child):
            termination_started.set()
            self._terminate(child)

        def remove_workspace():
            self.assertFalse(proc.alive, "stream cleanup hid a live oneshot")
            cleanup_started.set()

        async def stop_in_loop():
            self._stop(client)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                await stream.__anext__()

        with patch.object(client, "_cancel_owned_requests", delayed_cancel_cleanup, create=True), \
                patch.object(client, "_terminate_process", gated_terminate), \
                patch.object(client, "_remove_temp_dir", remove_workspace):
            try:
                asyncio.run(stop_in_loop())
                self.assertTrue(termination_started.wait(1))
                owner = threading.Thread(target=client.close, daemon=True)
                owner.start()
                self.assertFalse(cleanup_started.wait(0.05))
                proc.terminate_gate.set()
                owner.join(1)
                self.assertFalse(owner.is_alive())
                self.assertTrue(cleanup_started.is_set())
            finally:
                proc.terminate_gate.set()
                cancel_gate.set()
            self.assertTrue(cancel_done.wait(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
