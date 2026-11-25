"""A silent ``agy`` process is terminated instead of parking the read (issue #65).

``agy`` can open its upstream request and then emit nothing on stdout and
close nothing. ``readline()`` then blocks until the whole read deadline: 300 s
with the plugin default, 1800 s under the gateway's default timeout. The
existing watchdog already terminated the process on early quota errors; it now
also terminates it after ``ANTIGRAVITY_STREAM_STALL_SECONDS`` without a line.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stream as stream_module  # noqa: E402
from stream import (  # noqa: E402
    _DEFAULT_STREAM_STALL_SECONDS,
    _STREAM_STALL_ENV,
    AntigravityStream,
    _stream_stall_seconds,
)


class _HangingProc:
    """stdout yields ``lines`` and then stays silent until the process is terminated."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._gone = threading.Event()
        self.terminated = False
        self.stderr = None
        self.stdin = None
        self.stdout = self

    def poll(self):
        return 0 if self._gone.is_set() else None

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._gone.wait(timeout=8)
        return ""

    def terminate(self):
        self._gone.set()

    def wait(self, timeout=None):
        self._gone.wait(timeout=timeout)
        return 0


class _Client:
    """A client stand-in whose every other attribute is a mock: the stream's
    close path touches several of them and they must not mask the error under test."""

    def __init__(self, proc: _HangingProc) -> None:
        self._mock = MagicMock()
        self._mock._isolated_gemini_dir = None
        self._mock._terminate_process.side_effect = lambda p: setattr(proc, "terminated", True) or proc.terminate()
        self._proc = proc

    def __getattr__(self, name):
        return getattr(self._mock, name)


def _line(event: dict) -> str:
    return json.dumps(event) + "\n"


class StreamStallTests(unittest.TestCase):
    def setUp(self) -> None:
        # The watchdog only judges after its first second, so the budget sits below it.
        patcher = patch.dict("os.environ", {_STREAM_STALL_ENV: "0.3"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stream(self, proc: _HangingProc):
        client = _Client(proc)
        return AntigravityStream(proc=proc, client=client, model="fixture-only", timeout=5.0), client

    def test_silence_since_start_terminates_the_process_and_fails_the_turn(self) -> None:
        proc = _HangingProc([])  # not even an init event
        stream, _client = self._stream(proc)
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            list(stream)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertTrue(proc.terminated)
        self.assertIn("agy silent for", str(ctx.exception))
        self.assertIn("since start", str(ctx.exception))
        self.assertIn(f"{_STREAM_STALL_ENV}=0.3", str(ctx.exception))
        self.assertIn("process terminated", str(ctx.exception))

    def test_silence_after_output_keeps_the_yielded_text_and_fails_the_turn(self) -> None:
        proc = _HangingProc([
            _line({"event": "init", "conversation_id": "c1"}),
            _line({"event": "step_update", "step_update": {"text_delta": "partial answer"}}),
        ])
        stream, _client = self._stream(proc)
        chunks = []
        with self.assertRaises(RuntimeError) as ctx:
            for chunk in stream:
                chunks.append(chunk)
        text = "".join(
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        )
        self.assertEqual(text, "partial answer")
        self.assertIn("after its last output", str(ctx.exception))
        self.assertTrue(proc.terminated)
        self.assertFalse([c for c in chunks if c.choices and c.choices[0].finish_reason])

    def test_closing_an_abandoned_stream_stops_the_watchdog(self) -> None:
        # A budget the parked generator never reaches: only close() ends this thread.
        with patch.dict("os.environ", {_STREAM_STALL_ENV: "30"}):
            self._abandoned_stream_close_stops_the_watchdog()

    def _abandoned_stream_close_stops_the_watchdog(self) -> None:
        proc = _HangingProc([_line({"event": "init", "conversation_id": "c1"})])
        stream, _client = self._stream(proc)
        errors: list[BaseException] = []

        def consume() -> None:
            try:
                for _ in stream:
                    pass
            except BaseException as exc:  # the abandoned generator fails after close
                errors.append(exc)

        threading.Thread(target=consume, daemon=True).start()
        time.sleep(0.3)  # only the init event: the generator parks in readline
        done = stream._stall_done
        self.assertIsNotNone(done)
        stream.close()
        self.assertTrue(done.wait(timeout=2.0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and [t for t in threading.enumerate() if t.name == "agy-stall-watchdog"]:
            time.sleep(0.01)
        self.assertFalse([t for t in threading.enumerate() if t.name == "agy-stall-watchdog"])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not errors:
            time.sleep(0.01)
        self.assertTrue(errors, "the parked generator was released, not orphaned")  # noqa: PT018

    def test_zero_disables_the_watchdog(self) -> None:
        with patch.dict("os.environ", {_STREAM_STALL_ENV: "0"}):
            proc = _HangingProc([])  # exits on its own: nothing to wait for
            proc.terminate()
            stream, _client = self._stream(proc)
            with self.assertRaises(RuntimeError) as ctx:
                list(stream)
        self.assertIsNone(stream._stall_done)  # no thread was started
        self.assertIn("stdout closed", str(ctx.exception))
        self.assertNotIn("agy silent", str(ctx.exception))

    def test_budget_parsing(self) -> None:
        for raw, expected in (("", _DEFAULT_STREAM_STALL_SECONDS), ("120", 120.0), ("0", 0.0), ("-5", 0.0), ("soon", _DEFAULT_STREAM_STALL_SECONDS)):
            with self.subTest(raw=raw), patch.dict("os.environ", {_STREAM_STALL_ENV: raw}):
                self.assertEqual(_stream_stall_seconds(), expected)
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop(_STREAM_STALL_ENV, None)
            self.assertEqual(_stream_stall_seconds(), _DEFAULT_STREAM_STALL_SECONDS)


if __name__ == "__main__":
    unittest.main()
