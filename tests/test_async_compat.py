"""Async-compat regression tests for the Hermes async auxiliary wire (issue #8).

Hermes' async auxiliary client does ``await client.chat.completions.create(...)``
even for clients that declare ``HERMES_SKIP_ASYNC_WRAP`` (the "already
async-safe" contract), and its forced-stream plan does ``chunks = await
create(**stream_kwargs)`` followed by ``async for chunk in chunks``.
AntigravityClient.create used to return a plain ``SimpleNamespace`` / a
sync-only iterator, so every vision/aux call died with
``TypeError: object SimpleNamespace can't be used in 'await' expression``.

The fix is an awaitable boundary: ``collect_stream_completion`` returns a
``_AwaitableCompletion`` (awaitable, yields itself) and ``AntigravityStream``
gains ``__await__``/``__aiter__``/``__anext__`` (one subprocess read per
await, off the event loop). Sync callers are unchanged and pinned here as a
regression guard.

The suite is run with Hermes source on the path, as in CI:
    PYTHONPATH=/usr/local/lib/hermes-agent:. python3 -m pytest tests/ -q
The GOLD tests import the real ``agent.auxiliary_client._acreate_with_progress``
and drive both of its plans through an AntigravityClient with a mocked
subprocess; they skip (never silently pass) when that module is unavailable.
"""

import asyncio
import inspect
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from stream import AntigravityStream, _AwaitableCompletion

try:
    from agent.auxiliary_client import _acreate_with_progress, aux_progress_hook
except ImportError as exc:  # host without the Hermes source tree on sys.path
    _HERMES_AUX_IMPORT_ERROR: Exception | None = exc
else:
    _HERMES_AUX_IMPORT_ERROR = None

MODEL = "gemini-3.8-flash-high"
TASK = "vision"  # the aux task of the reported failure
MESSAGES = [{"role": "user", "content": "turn one"}]
MESSAGES_TURN_2 = MESSAGES + [
    {"role": "assistant", "content": "answer one"},
    {"role": "user", "content": "turn two"},
]

TURN_1_USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cache_read_tokens": 0}
TURN_2_USAGE = {"input_tokens": 250, "output_tokens": 40, "total_tokens": 290, "cache_read_tokens": 0}

# Off-loop test margins: a chunk whose readline blocks for 0.25s must not
# delay a concurrent 0.05s sleeper (5x, comfortably over the 4x floor).
_CHUNK_LATENCY_SECONDS = 0.25
_SLEEPER_SECONDS = 0.05


async def _await_directly(value: Any) -> Any:
    """Await *value* in the caller's own coroutine frame."""
    return await value


async def _await_via_task(value: Any) -> Any:
    """Await *value* inside a dedicated asyncio Task.

    A Task given a non-Future yield crashes with "Task got bad yield", so this
    harness pins ``__await__`` against the tempting-but-broken
    ``iter([self])`` shortcut, in the same position Hermes awaits our result.
    """
    return await asyncio.create_task(_await_directly(value))


def _await_plain(value: Any) -> Any:
    return asyncio.run(_await_directly(value))


def _await_in_task(value: Any) -> Any:
    return asyncio.run(_await_via_task(value))


async def _consume_stream_async(value: Any) -> tuple[Any, list[Any]]:
    """Mirror the Hermes async wire: ``chunks = await create(...)`` then ``async for``."""
    chunks = await value
    collected: list[Any] = []
    async for chunk in chunks:
        collected.append(chunk)
    return chunks, collected


def _consume_stream_sync(stream: Any) -> list[Any]:
    """Mirror the pre-existing sync wire: plain ``for`` over the same object."""
    return list(stream)


def _turn_events(conversation_id: str, text: str, usage: dict) -> list[dict]:
    """stream-json events a single agy worker turn emits before its usage."""
    return [
        {"event": "init", "conversation_id": conversation_id},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {"event": "result", "result": {"status": "SUCCESS", "response": text, "usage": usage}},
    ]


def _lines_for(*turns: list[dict]) -> list[str]:
    lines: list[str] = []
    for turn in turns:
        lines.extend(json.dumps(event) + "\n" for event in turn)
    lines.append("")
    return lines


def _mock_proc(lines: list[str], *, alive: bool = True) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None if alive else 0
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = lines
    return proc


def _slow_readline(lines: list[str], latency: float):
    """``readline`` stand-in that really blocks *latency* seconds per call.

    A mock ``side_effect`` list answers instantly, so only a real sleep makes
    the generator's blocking read observable from the event loop.
    """
    pending = iter(lines)

    def _readline() -> str:
        time.sleep(latency)
        return next(pending, "")

    return _readline


def _completion_fields(res: Any) -> dict:
    """Field-by-field snapshot used to compare sync and awaited results."""
    message = res.choices[0].message
    return {
        "id": res.id,
        "model": res.model,
        "role": message.role,
        "content": message.content,
        "tool_calls": message.tool_calls,
        "finish_reason": res.choices[0].finish_reason,
        "usage": (
            res.usage.prompt_tokens,
            res.usage.completion_tokens,
            res.usage.total_tokens,
            res.usage.prompt_tokens_details.cached_tokens,
        ),
    }


def _stream_content(chunks: list[Any]) -> list[str]:
    return [
        choice.delta.content
        for chunk in chunks
        for choice in getattr(chunk, "choices", [])
        if getattr(getattr(choice, "delta", None), "content", None)
    ]


class _PinnedSeamsMixin:
    """Pin the same host-dependent seams as the rest of the suite."""

    def setUp(self):
        patcher_auth = patch("client.is_authenticated", return_value=True)
        patcher_token = patch("process.resolve_real_token_path", return_value=None)
        patcher_cmd = patch("client.resolve_agy_command", return_value="agy")
        patcher_auth.start()
        patcher_token.start()
        patcher_cmd.start()
        self.addCleanup(patcher_auth.stop)
        self.addCleanup(patcher_token.stop)
        self.addCleanup(patcher_cmd.stop)

    def _client(self) -> AntigravityClient:
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(client.close)
        return client

    def _turn_lines(self, usage: dict = TURN_1_USAGE) -> list[str]:
        return _lines_for(_turn_events("conv-1", "answer one", usage))


class AwaitableProtocolTests(_PinnedSeamsMixin, unittest.TestCase):
    """Unit checks on the awaitable protocol itself."""

    def test_nonstreaming_completion_is_awaitable_and_yields_itself(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            res = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False)
        self.assertIsInstance(res, _AwaitableCompletion)
        self.assertTrue(inspect.isawaitable(res))
        awaited = _await_plain(res)
        # Awaiting must hand back the very same object, never a re-executed copy.
        self.assertIs(awaited, res)

    def test_nonstreaming_completion_awaited_inside_task_yields_itself(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            res = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False)
        self.assertIs(_await_in_task(res), res)

    def test_awaited_completion_keeps_attribute_shape(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            res = _await_plain(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False))
        # The attribute shape must stay identical to the pre-fix SimpleNamespace.
        self.assertEqual(set(vars(res)), {"id", "choices", "usage", "model"})
        self.assertEqual(res.id, "conv-1")
        self.assertEqual(res.choices[0].message.content, "answer one")

    def test_stream_is_awaitable_and_yields_itself_without_consuming(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        self.assertIsInstance(stream, AntigravityStream)
        self.assertTrue(inspect.isawaitable(stream))
        awaited = _await_in_task(stream)
        self.assertIs(awaited, stream)
        # Awaiting must not consume chunks: the sync iteration below is complete.
        self.assertEqual(_stream_content(_consume_stream_sync(stream)), ["answer one"])

    def test_anext_on_exhausted_stream_raises_stop_async_iteration(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        _consume_stream_sync(stream)
        with self.assertRaises(StopAsyncIteration):
            asyncio.run(_await_directly(stream.__anext__()))

    def test_anext_maps_stop_iteration_on_a_closed_stream(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        stream.close()
        with self.assertRaises(StopAsyncIteration):
            asyncio.run(_await_directly(stream.__anext__()))

    def test_pull_chunk_reports_end_of_stream_as_a_value(self):
        # Pins the thread-boundary contract of _pull_chunk: StopIteration must
        # NOT escape into asyncio.to_thread (it cannot be raised into a Future
        # and would wedge the awaiting coroutine), so end of stream is
        # reported as (True, None) and mapped by __anext__.
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        exhausted, chunk = stream._pull_chunk()
        self.assertFalse(exhausted)
        self.assertEqual(chunk.choices[0].delta.content, "answer one")
        _consume_stream_sync(stream)
        self.assertEqual(stream._pull_chunk(), (True, None))


class AwaitedCreateWireTests(_PinnedSeamsMixin, unittest.TestCase):
    """Drive the real client facade over both create paths, sync vs awaited."""

    def test_awaited_nonstreaming_worker_matches_sync(self):
        sync_client = self._client()
        async_client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            sync_res = sync_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False)
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            async_res = _await_plain(async_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False))
        self.assertEqual(_completion_fields(sync_res), _completion_fields(async_res))
        self.assertEqual(async_res.choices[0].message.content, "answer one")

    def test_awaited_nonstreaming_oneshot_matches_sync(self):
        sync_client = self._client()
        async_client = self._client()
        with patch("subprocess.Popen", side_effect=[_mock_proc(self._turn_lines(), alive=False)]):
            self.assertTrue(sync_client._worker_lock.acquire(blocking=False))
            try:
                sync_res = sync_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False)
            finally:
                sync_client._worker_lock.release()
        with patch("subprocess.Popen", side_effect=[_mock_proc(self._turn_lines(), alive=False)]):
            self.assertTrue(async_client._worker_lock.acquire(blocking=False))
            try:
                async_res = _await_plain(async_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False))
            finally:
                async_client._worker_lock.release()
        self.assertEqual(_completion_fields(sync_res), _completion_fields(async_res))

    def test_awaited_stream_async_for_matches_sync_iteration(self):
        sync_client = self._client()
        async_client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = sync_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            sync_chunks = _consume_stream_sync(stream)
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            _, async_chunks = asyncio.run(
                _consume_stream_async(async_client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
            )
        self.assertEqual(_stream_content(sync_chunks), _stream_content(async_chunks))
        self.assertEqual(_stream_content(async_chunks), ["answer one"])
        # The trailing usage chunk survives the async aggregation intact.
        self.assertEqual(
            (async_chunks[-1].usage.prompt_tokens, async_chunks[-1].usage.completion_tokens),
            (100, 10),
        )

    def test_awaited_stream_object_is_the_created_stream(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            chunks, collected = asyncio.run(_consume_stream_async(stream))
        self.assertIs(chunks, stream)
        self.assertEqual(_stream_content(collected), ["answer one"])

    def test_sync_for_loop_regression_after_awaitable_change(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            contents = [chunk.choices[0].delta.content for chunk in stream if chunk.choices]
        self.assertEqual([c for c in contents if c], ["answer one"])

    def test_awaited_nonstreaming_regression_against_sync_loop(self):
        # The pre-fix sync call site (`res = create(...)` then attribute access)
        # must keep working untouched on the same client that is also awaited.
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            sync_res = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False)
            self.assertEqual(sync_res.choices[0].message.content, "answer one")
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            awaited_res = _await_plain(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=False))
        self.assertEqual(_completion_fields(sync_res), _completion_fields(awaited_res))


class AsyncStreamSemanticsTests(_PinnedSeamsMixin, unittest.TestCase):
    """Lock/error semantics of the async iteration path.

    ``__anext__`` runs ``__next__`` in a worker thread, so the generator's
    ``finally`` (which releases the worker lock and closes the stream) also
    runs in that thread. ``threading.Lock`` is not owner-bound, but the
    release must still happen before the next turn's ``acquire(blocking=False)``
    in ``_create_chat_completion``, or the next turn silently falls back to a
    fresh oneshot process (a second Popen) — the pin below.
    """

    def test_worker_lock_released_between_async_turns(self):
        client = self._client()
        worker_proc = _mock_proc(
            _lines_for(
                _turn_events("conv-1", "answer one", TURN_1_USAGE),
                _turn_events("conv-1", "answer two", TURN_2_USAGE),
            )
        )
        with patch("subprocess.Popen", return_value=worker_proc) as popen:
            _, chunks1 = asyncio.run(
                _consume_stream_async(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
            )
            self.assertFalse(client._worker_lock.locked())

            # Second worker turn must reuse the same worker process: had the
            # release above not happened, the non-blocking acquire would fail
            # and Popen would be called a second time for an oneshot process.
            _, chunks2 = asyncio.run(
                _consume_stream_async(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_2, stream=True)
                )
            )
            self.assertFalse(client._worker_lock.locked())
            self.assertEqual(popen.call_count, 1)
            self.assertIs(client._worker_proc, worker_proc)
        self.assertEqual(_stream_content(chunks1), ["answer one"])
        self.assertEqual(_stream_content(chunks2), ["answer two"])
        # Worker accounting still deltas against the advanced baseline.
        self.assertEqual(
            (chunks2[-1].usage.prompt_tokens, chunks2[-1].usage.completion_tokens),
            (150, 30),
        )

    def test_anext_propagates_stream_error_after_close(self):
        client = self._client()
        error_lines = [
            json.dumps({"event": "init", "conversation_id": "conv-1"}) + "\n",
            json.dumps({"event": "result", "result": {"status": "ERROR", "error": "boom"}}) + "\n",
            "",
        ]
        with patch("subprocess.Popen", return_value=_mock_proc(error_lines)):
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)

            async def _consume():
                await stream
                async for _chunk in stream:
                    pass

            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(_consume())
        self.assertIn("boom", str(ctx.exception))
        # The interrupted path released the worker lock and terminated the worker.
        self.assertFalse(client._worker_lock.locked())
        self.assertIsNone(client._worker_proc)

    def test_concurrent_worker_and_oneshot_streams_in_one_event_loop(self):
        # Two streams alive in the SAME event loop, consumed concurrently
        # through __anext__ (each chunk pulled via to_thread): the worker
        # stream holds the worker lock, so the concurrent request must fall
        # back to its own oneshot process, and both must complete with the
        # lock released and the oneshot process cleaned afterwards. A release
        # that never happened (or deadlocked) shows up as a second Popen for
        # the worker turn or as a hung test.
        client = self._client()
        worker_proc = _mock_proc(_lines_for(_turn_events("conv-w", "worker answer", TURN_1_USAGE)))
        oneshot_proc = _mock_proc(
            _lines_for(_turn_events("conv-o", "oneshot answer", TURN_2_USAGE)), alive=False
        )
        out: dict[str, list[str]] = {}

        async def _consume(name: str, stream: Any) -> None:
            contents: list[str] = []
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                delta = choices[0].delta if choices else None
                if delta is not None and delta.content:
                    contents.append(delta.content)
            out[name] = contents

        async def _drive() -> None:
            worker_stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            self.assertTrue(client._worker_lock.locked())
            oneshot_stream = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "other"}], stream=True
            )
            await asyncio.gather(_consume("worker", worker_stream), _consume("oneshot", oneshot_stream))

        with patch("subprocess.Popen", side_effect=[worker_proc, oneshot_proc]):
            asyncio.run(_drive())
        self.assertEqual(out, {"worker": ["worker answer"], "oneshot": ["oneshot answer"]})
        self.assertFalse(client._worker_lock.locked())
        self.assertIs(client._worker_proc, worker_proc)
        self.assertNotIn(oneshot_proc, client._active_processes)

    def test_chunk_reads_run_off_the_event_loop(self):
        # Core design claim of the async stream path: each blocking subprocess
        # read happens in a worker thread (asyncio.to_thread inside __anext__),
        # so the event loop keeps running while a chunk is in flight. A
        # concurrent 0.05s sleeper must finish BEFORE a chunk whose readline
        # blocks 0.25s (5x margin, over the 4x floor) arrives; a __anext__
        # that read inline could only release the sleeper after the chunk.
        client = self._client()
        proc = _mock_proc([])
        proc.stdout.readline.side_effect = _slow_readline(
            _lines_for(_turn_events("conv-1", "answer one", TURN_1_USAGE)),
            _CHUNK_LATENCY_SECONDS,
        )
        order: list[str] = []

        async def _drive() -> None:
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)

            async def _sleeper() -> None:
                await asyncio.sleep(_SLEEPER_SECONDS)
                order.append("sleeper")

            async def _first_chunk() -> None:
                async for _chunk in stream:
                    order.append("chunk")
                    break

            await asyncio.gather(_sleeper(), _first_chunk())

        with patch("subprocess.Popen", return_value=proc):
            asyncio.run(_drive())
        self.assertEqual(order, ["sleeper", "chunk"])

    def test_concurrent_anext_on_one_stream_fails_loudly(self):
        # Re-entrance parity with the sync path: two threads calling next() on
        # the same iterator make the loser raise ValueError("generator already
        # executing"). Two concurrent __anext__ awaits on one live stream must
        # reproduce exactly that loud failure (one chunk, one ValueError) --
        # this pin is deliberate. A future edit that makes __anext__
        # re-entrance-tolerant must fail this test and be a conscious
        # decision, never an accident of a refactor.
        client = self._client()
        # The slow readline keeps the first __anext__ inside the generator
        # long enough for the second to hit the non-reentrant next(), so the
        # outcome cannot depend on thread scheduling.
        proc = _mock_proc([])
        proc.stdout.readline.side_effect = _slow_readline(
            _lines_for(_turn_events("conv-1", "answer one", TURN_1_USAGE)),
            _CHUNK_LATENCY_SECONDS,
        )

        async def _drive() -> list[Any]:
            stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            return await asyncio.gather(
                stream.__anext__(), stream.__anext__(), return_exceptions=True
            )

        with patch("subprocess.Popen", return_value=proc):
            results = asyncio.run(_drive())
        chunks = [r for r in results if not isinstance(r, BaseException)]
        errors = [r for r in results if isinstance(r, BaseException)]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertIn("generator already executing", str(errors[0]))


@unittest.skipIf(
    _HERMES_AUX_IMPORT_ERROR is not None,
    f"agent.auxiliary_client not importable ({_HERMES_AUX_IMPORT_ERROR}); "
    "run with PYTHONPATH=/usr/local/lib/hermes-agent:.",
)
class HermesAuxAsyncWireTests(_PinnedSeamsMixin, unittest.TestCase):
    """GOLD: the real Hermes async auxiliary wire against this plugin.

    Both plans of ``_acreate_with_progress`` are driven with the await intact
    (nothing patches it out): the plain non-stream plan and the forced-stream
    plan, plus the progress-hook plan Hermes itself uses while compressing.
    With the ``__await__`` methods reverted these tests fail with the original
    ``TypeError: object ... can't be used in 'await' expression``.
    """

    @staticmethod
    def _kwargs() -> dict:
        """Aux request kwargs as Hermes' vision/aux path sends them (no stream)."""
        return {"model": MODEL, "messages": MESSAGES, "timeout": 60}

    def test_acreate_with_progress_nonstream_plan(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            res = asyncio.run(_acreate_with_progress(client, self._kwargs(), TASK))
        self.assertEqual(res.choices[0].message.content, "answer one")
        self.assertEqual(res.choices[0].finish_reason, "stop")
        self.assertEqual(res.choices[0].message.role, "assistant")
        self.assertEqual(res.usage.prompt_tokens, 100)
        self.assertEqual(res.usage.completion_tokens, 10)
        self.assertEqual(res.usage.total_tokens, 110)

    def test_acreate_with_progress_forced_stream_plan(self):
        client = self._client()
        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            res = asyncio.run(_acreate_with_progress(client, self._kwargs(), TASK, force_stream=True))
        self.assertEqual(res.choices[0].message.content, "answer one")
        self.assertEqual(res.choices[0].finish_reason, "stop")
        self.assertEqual(res.usage.prompt_tokens, 100)
        self.assertEqual(res.usage.completion_tokens, 10)
        self.assertEqual(res.usage.total_tokens, 110)

    def test_acreate_with_progress_streams_and_ticks_progress_with_hook(self):
        client = self._client()
        ticks: list[int] = []

        def _tick_progress() -> None:
            ticks.append(1)

        with patch("subprocess.Popen", return_value=_mock_proc(self._turn_lines())):
            with aux_progress_hook(_tick_progress):
                res = asyncio.run(_acreate_with_progress(client, self._kwargs(), TASK))
        # An active progress hook switches Hermes to the streamed plan, whose
        # per-chunk ticks are the whole point of that plan.
        self.assertTrue(ticks)
        self.assertEqual(res.choices[0].message.content, "answer one")
        self.assertEqual(res.usage.prompt_tokens, 100)


if __name__ == "__main__":
    unittest.main()
