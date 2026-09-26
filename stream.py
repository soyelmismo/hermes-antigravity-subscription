"""Streaming iterator and response assembly for Antigravity stream-json events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Iterator, NamedTuple

try:
    from .process import _check_early_quota_error
    from .prompt import _longest_tool_call_prefix_match, _parse_tool_block
except ImportError:
    from process import _check_early_quota_error
    from prompt import _longest_tool_call_prefix_match, _parse_tool_block

logger = logging.getLogger(__name__)


class _UsageDeltas(NamedTuple):
    """Per-turn usage derived from cumulative worker-session counters."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int


def _counter_delta(usage_data: dict[str, Any], baseline: dict[str, int], field: str) -> int:
    """Per-turn delta of one cumulative usage counter reported by a worker.

    agy 1.2.10+ persistent workers report cumulative session usage, so each
    turn must subtract the snapshot captured after the previous turn. The
    baseline is updated in place with the latest snapshot. A missing field
    contributes 0 and leaves the baseline untouched, so an omitted optional
    field never resets unrelated counters. A value below the baseline means
    the CLI restarted its counters for this session: the current value is
    already this turn's usage and becomes the new baseline.

    Known limitations of the remaining handling (kept on purpose; no code
    change, no better signal exists in the delta alone):

    * A value below the baseline is a SUFFICIENT but not a definitive restart
      detector. A CLI restart whose counter already reports >= the old
      baseline reads as a small positive delta: the pre-restart usage is
      attributed to that (in fact empty) turn instead of restarting the
      accounting.
    * A counter first reported mid-session (no previous snapshot, e.g.
      cache_read_tokens appearing only once the prompt cache warms) is
      attributed in full to the current turn. How that session-to-date value
      splits across earlier turns is unknowable, so this turn over-reports.
    """
    if field not in usage_data:
        return 0
    value = int(usage_data[field] or 0)
    previous = baseline.get(field)
    if previous is None or value < previous:
        baseline[field] = value
        return value
    delta = value - previous
    baseline[field] = value
    return delta


def _delta_worker_usage(usage_data: dict[str, Any], baseline: dict[str, int]) -> _UsageDeltas:
    """Convert cumulative worker-session usage into per-turn deltas.

    The baseline dict belongs to the worker session that produced this usage
    and is updated in place. Callers must hand in the baseline captured when
    the worker stream was created, so a stream left over from a terminated
    session can never corrupt the baseline of the session that replaced it.

    total_tokens is expected to be stably present or absent for a session. If
    it flaps (present -> absent -> present), the absent turn falls back to
    input+output without advancing the total_tokens snapshot, so the next
    present turn's total delta spans two turns.
    """
    input_delta = _counter_delta(usage_data, baseline, "input_tokens")
    output_delta = _counter_delta(usage_data, baseline, "output_tokens")
    if "total_tokens" in usage_data:
        total_delta = _counter_delta(usage_data, baseline, "total_tokens")
    else:
        # total_tokens is optional: fall back to the per-turn input+output.
        total_delta = input_delta + output_delta
    return _UsageDeltas(
        input_tokens=input_delta,
        output_tokens=output_delta,
        total_tokens=total_delta,
        cache_read_tokens=_counter_delta(usage_data, baseline, "cache_read_tokens"),
    )


async def _ready(value: Any) -> Any:
    """Trivial coroutine that immediately returns *value*.

    The return channel of every ``__await__`` in this module: awaiting an
    already-completed result must produce that result without doing any work
    (no re-execution, no re-reading of the subprocess).
    """
    return value


class _AwaitableCompletion(SimpleNamespace):
    """A completed ChatCompletion that is also awaitable, yielding itself.

    Hermes' async auxiliary path awaits ``create()`` even for clients that
    declare ``HERMES_SKIP_ASYNC_WRAP`` ("already async-safe"): the plugin is
    expected to return something legal to ``await`` from its synchronous,
    already-finished round-trip. A plain ``SimpleNamespace`` is not awaitable,
    which surfaced as ``TypeError: object SimpleNamespace can't be used in
    'await' expression`` on vision/auxiliary calls (issue #8).

    The attribute shape is exactly ``SimpleNamespace``'s
    (``.id/.choices/.usage/.model``); only awaitability is added, so sync
    callers see byte-identical objects.
    """

    def __await__(self) -> Any:
        # Yield a coroutine that immediately returns self. Deliberately NOT
        # `iter([self])`: a Task receiving a non-Future yield crashes with
        # "Task got bad yield", so the yield must come from a real coroutine.
        return _ready(self).__await__()


class AntigravityStream(Iterator[Any]):
    """Streaming iterator yielding OpenAI ChatCompletionChunk objects from CLI stream-json."""

    response: Any = None  # Mock response attribute for Hermes Relay compatibility

    def __init__(
        self,
        *,
        proc: subprocess.Popen,
        client: Any,
        model: str,
        timeout: float,
        tools: list[dict[str, Any]] | None = None,
        is_worker: bool = False,
        worker_lock: threading.Lock | None = None,
        messages: list[dict[str, Any]] | None = None,
        usage_baseline: dict[str, int] | None = None,
    ):
        self.proc = proc
        self.client = client
        self.model = model
        self.timeout = timeout
        self.has_tools = bool(tools)
        self.is_worker = is_worker
        self.worker_lock = worker_lock
        self.messages = messages
        # Cumulative usage snapshot of the worker session that owns this
        # stream; only worker streams receive one (see client._create_chat_completion).
        self.usage_baseline = usage_baseline
        self.conversation_id = ""
        self._closed = False
        self._interrupted = False
        self._finished = False
        self._early_error: str | None = None
        self._generator = self._stream_generator()
        # Private single-thread executor for the async path, created lazily on
        # the first __anext__ (sync consumers never create one, and a closed
        # reference after close() is what keeps a stale submit loud). See
        # __anext__ for why this is not the shared default executor.
        self._async_executor: ThreadPoolExecutor | None = None

    def __iter__(self) -> "AntigravityStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration:
            self._closed = True
            if not self.is_worker:
                self.close()
            raise
        except Exception:
            self.close()
            raise

    def __await__(self) -> Any:
        """Allow ``chunks = await create(stream=True)`` on the async wire.

        Same contract as :class:`_AwaitableCompletion`: the stream (and the
        subprocess round-trip behind it) is already fully constructed, so
        awaiting yields the stream itself without consuming any chunk;
        consumption goes through ``__aiter__``/``__anext__`` below.
        """
        return _ready(self).__await__()

    def __aiter__(self) -> "AntigravityStream":
        return self

    def _pull_chunk(self) -> tuple[bool, Any]:
        """Run one sync ``__next__``; ``(True, None)`` means the stream is done.

        End of stream must be reported as a *value*: StopIteration cannot be
        raised into a Future (asyncio and ``concurrent.futures`` reject it
        with "StopIteration interacts badly with generators"), so letting it
        escape the worker thread would wedge the awaiting coroutine forever
        instead of ending the ``async for``. Runs on this stream's private
        executor (see ``__anext__``), so a blocking readline occupies one
        private thread for at most the request timeout.
        """
        try:
            return False, self.__next__()
        except StopIteration:
            return True, None

    async def __anext__(self) -> Any:
        """One blocking subprocess read per await, off the event loop.

        Each ``next()`` runs on this stream's PRIVATE single-thread executor
        (created lazily on the first async pull; sync consumers never create
        one). The shared default executor is deliberately not used: one
        ``readline()`` can occupy its thread for the whole request timeout
        (300s by default), and starving every other ``run_in_executor(None,
        ...)`` user in the host loop for that long is not acceptable.

        The closed check comes FIRST, before the executor is touched: after
        ``close()`` the executor is shut down, and submitting to a dead
        executor raises RuntimeError -- a closed stream is simply done, so
        ``StopAsyncIteration`` is the caller-visible outcome.

        The worker-side stop sentinel from ``_pull_chunk`` is mapped back to
        ``StopAsyncIteration`` so ``async for`` terminates exactly like the
        sync ``for`` loop; every other exception propagates unchanged, after
        the same ``close()`` the sync path performs.

        Known limitation (documented, deliberately not fixed): awaiting the
        stream object itself only yields the stream, so a consumer still
        drives the subprocess chunk by chunk from worker threads. Per-chunk,
        not per-request, non-blocking is the best a plugin can do without a
        Hermes hook that would hand out a coroutine from create().

        Cancelling an in-flight ``__anext__`` (e.g. a ``wait_for`` timeout)
        leaves the stream unclosed: the worker lock stays held, the worker
        process stays alive, and the private executor's worker stays blocked
        in ``readline()`` until something calls ``close()``. Hermes' own
        ``_aggregate_chat_stream_async`` supplies that ``close()`` on
        realistic paths -- its ``finally`` runs ``_close_chunk_stream(chunks,
        allow_aclose=True)``, which finds this class's sync ``close()`` --
        matching the sync behavior of abandoning a stream without calling
        ``close()``. No ``CancelledError`` handler is added here on purpose:
        reacting to cancellation would change the close/terminate semantics,
        which is out of scope for this fix. A stream neither consumed nor
        closed keeps its executor thread (non-daemon, like every
        ThreadPoolExecutor thread) blocked until the process exits; the same
        is true of an abandoned sync stream's subprocess, and closing the
        client terminates the child, unblocking the read.
        """
        if self._closed:
            raise StopAsyncIteration
        executor = self._async_executor
        if executor is None:
            executor = self._async_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="agy-stream"
            )
        stream_exhausted, chunk = await asyncio.get_running_loop().run_in_executor(
            executor, self._pull_chunk
        )
        if stream_exhausted:
            raise StopAsyncIteration
        return chunk

    async def aclose(self) -> None:
        """Async close for ``async with contextlib.aclosing(stream)`` consumers.

        Deliberately synchronous inside (``self.close()``). Reasons: Hermes'
        own ``_close_chunk_stream(chunks, allow_aclose=True)`` prefers the
        plain ``close`` attribute anyway, so this exists for direct-await
        consumers and ``aclosing``, not for the Hermes wire; close() is
        bounded (process terminate + ``wait(2)`` + kill fallback); routing it
        through the DEFAULT executor would re-introduce the shared-executor
        occupancy that ``__anext__`` avoids, and routing it through THIS
        stream's executor would deadlock when close() runs on that very
        thread (see ``close``).
        """
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.is_worker:
            if not self._finished:
                self._interrupted = True
                self.client._terminate_worker()
            if self.worker_lock and self.worker_lock.locked():
                with contextlib.suppress(Exception):
                    self.worker_lock.release()
        else:
            with self.client._lock:
                self.client._active_processes.discard(self.proc)
            self.client._terminate_process(self.proc)
        # Retire the private async executor, if the async path created one.
        # wait=False is mandatory: close() also runs ON the executor thread
        # (the __next__ error path reaches it from _pull_chunk), so joining
        # would deadlock on the caller. cancel_futures=False: an in-flight
        # _pull_chunk has already started and cannot be cancelled -- and it
        # does not need to be, because terminating the process above makes
        # its readline() return EOF, so the worker thread finishes on its
        # own. The executor reference is kept (not None'd) so a stale submit
        # after close() fails loudly instead of silently starting a new
        # executor; __anext__ guards that with its _closed check first.
        executor = self._async_executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

    def _make_chunk(
        self,
        *,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
            choices=[choice],
            model=self.model,
            usage=None,
        )

    def _usage_totals(self, usage_data: dict[str, Any]) -> tuple[int, int, int, int]:
        """(input, output, total, cached) token counts for the usage chunk.

        Persistent worker sessions report cumulative usage, so worker streams
        forward per-turn deltas against their session baseline. Oneshot
        processes already report per-turn usage and are forwarded raw; they
        never touch the worker baseline.
        """
        if self.is_worker and self.usage_baseline is not None:
            deltas = _delta_worker_usage(usage_data, self.usage_baseline)
            return (
                deltas.input_tokens,
                deltas.output_tokens,
                deltas.total_tokens,
                deltas.cache_read_tokens,
            )
        input_tokens = int(usage_data.get("input_tokens", 0) or 0)
        output_tokens = int(usage_data.get("output_tokens", 0) or 0)
        total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens) or 0)
        cached_tokens = int(usage_data.get("cache_read_tokens", 0) or 0)
        return input_tokens, output_tokens, total_tokens, cached_tokens

    def _stream_generator(self) -> Iterator[Any]:
        deadline = time.monotonic() + self.timeout
        text_buffer = ""
        usage_data: dict[str, Any] = {}
        has_tool_calls = False
        has_content = False
        in_tool_call = False
        error_msg = ""
        status = ""
        success = False
        start_time = time.monotonic()

        gemini_dir = getattr(self.client, "_isolated_gemini_dir", None)
        watchdog_stop = threading.Event()
        watchdog_thread: threading.Thread | None = None

        if gemini_dir:
            def _watchdog_loop() -> None:
                while not watchdog_stop.wait(timeout=0.3):
                    if time.monotonic() - start_time < 1.0:
                        continue
                    quota_err = _check_early_quota_error(gemini_dir, min_mtime=start_time)
                    if quota_err:
                        self._early_error = quota_err
                        logger.warning(
                            "Antigravity process hit early quota limit: %s; terminating to prevent hang.",
                            quota_err,
                        )
                        self.client._terminate_process(self.proc)
                        break

            watchdog_thread = threading.Thread(
                target=_watchdog_loop,
                name="agy-quota-watchdog",
                daemon=True,
            )
            watchdog_thread.start()

        try:
            while time.monotonic() < deadline:
                if self._early_error:
                    raise RuntimeError(f"Antigravity model error: {self._early_error}")

                line = self.proc.stdout.readline() if self.proc.stdout else ""
                if self._early_error:
                    raise RuntimeError(f"Antigravity model error: {self._early_error}")

                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue

                line = line.strip()
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                watchdog_stop.set()

                event_type = event.get("event")
                if not self.conversation_id:
                    self.conversation_id = event.get("conversation_id", "")

                if event_type == "init":
                    init_data = event.get("init", {})
                    if not self.conversation_id:
                        self.conversation_id = init_data.get("conversation_id", "")

                elif event_type == "step_update":
                    step = event.get("step_update", {})
                    if not self.conversation_id:
                        self.conversation_id = step.get("conversation_id", "")
                    if step.get("step_type") == "tool":
                        logger.warning(
                            "Antigravity attempted native tool invocation '%s'; neutralizing to prevent host execution.",
                            step.get("tool_name"),
                        )
                        self.close()
                        break
                    if "usage" in step:
                        usage_data = step["usage"]

                    text_delta = step.get("text_delta")
                    if text_delta:
                        if not self.has_tools:
                            has_content = True
                            yield self._make_chunk(content=text_delta)
                        else:
                            text_buffer += text_delta
                            while text_buffer:
                                if in_tool_call:
                                    end_idx = text_buffer.find("</tool_call>")
                                    if end_idx != -1:
                                        full_end = end_idx + len("</tool_call>")
                                        block = text_buffer[:full_end]
                                        text_buffer = text_buffer[full_end:]
                                        in_tool_call = False
                                        parsed_calls, extra_text = _parse_tool_block(block)
                                        if parsed_calls:
                                            has_tool_calls = True
                                            for call_delta in parsed_calls:
                                                yield self._make_chunk(tool_calls=[call_delta])
                                        if extra_text and not has_tool_calls:
                                            has_content = True
                                            yield self._make_chunk(content=extra_text)
                                    else:
                                        break
                                else:
                                    idx = text_buffer.find("<tool_call")
                                    if idx != -1:
                                        if idx > 0 and not has_tool_calls:
                                            safe_text = text_buffer[:idx]
                                            has_content = True
                                            yield self._make_chunk(content=safe_text)
                                        text_buffer = text_buffer[idx:]
                                        in_tool_call = True
                                    else:
                                        k = _longest_tool_call_prefix_match(text_buffer)
                                        if k > 0:
                                            safe_text = text_buffer[:-k]
                                            if safe_text and not has_tool_calls:
                                                has_content = True
                                                yield self._make_chunk(content=safe_text)
                                            text_buffer = text_buffer[-k:]
                                            break
                                        else:
                                            if not has_tool_calls:
                                                has_content = True
                                                yield self._make_chunk(content=text_buffer)
                                            text_buffer = ""

                elif event_type == "result":
                    res = event.get("result", {})
                    if not self.conversation_id:
                        self.conversation_id = res.get("conversation_id", "")
                    status = res.get("status", "")
                    if "usage" in res:
                        usage_data = res["usage"]
                    if "error" in res:
                        error_msg = res["error"]
                    success = (status != "ERROR")
                    final_resp = res.get("response", "")
                    if final_resp and not has_content and not has_tool_calls and not text_buffer:
                        text_buffer = final_resp
                    break

            if text_buffer and not has_tool_calls:
                if self.has_tools or "<tool_call>" in text_buffer:
                    parsed_calls, extra_text = _parse_tool_block(text_buffer)
                    if parsed_calls:
                        has_tool_calls = True
                        for call_delta in parsed_calls:
                            yield self._make_chunk(tool_calls=[call_delta])
                    if extra_text and not has_tool_calls:
                        has_content = True
                        yield self._make_chunk(content=extra_text)
                    elif not parsed_calls:
                        has_content = True
                        yield self._make_chunk(content=text_buffer)
                else:
                    has_content = True
                    yield self._make_chunk(content=text_buffer)

            if not self.is_worker:
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.client._terminate_process(self.proc)

                stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
                returncode = self.proc.poll() or 0

                if self._early_error:
                    raise RuntimeError(f"Antigravity model error: {self._early_error}")

                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")

                if not has_tool_calls and not has_content and returncode != 0:
                    quota_err = _check_early_quota_error(gemini_dir, min_mtime=start_time)
                    if quota_err:
                        raise RuntimeError(f"Antigravity model error: {quota_err}")
                    err_detail = error_msg or stderr_out.strip() or f"Process exited with return code {returncode}"
                    raise RuntimeError(f"Antigravity execution failed: {err_detail}")
            else:
                if self._early_error:
                    raise RuntimeError(f"Antigravity model error: {self._early_error}")
                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")
                if not has_tool_calls and not has_content and self.proc.poll() not in (None, 0):
                    quota_err = _check_early_quota_error(gemini_dir, min_mtime=start_time)
                    if quota_err:
                        raise RuntimeError(f"Antigravity model error: {quota_err}")
                    returncode = self.proc.poll()
                    raise RuntimeError(f"Antigravity execution failed: worker process exited with return code {returncode}")

            finish_reason = "tool_calls" if has_tool_calls else "stop"

            # Resolve usage (advancing the cumulative worker baseline) BEFORE
            # emitting the finish-reason chunk. The common client pattern
            # `for chunk in stream: if finish_reason: break` abandons the
            # stream right here without calling close(), finalizing this
            # generator at this yield: the usage chunk below is then never
            # received, but the turn's tokens were already spent, so the
            # baseline must not wait for that chunk or the next turn's delta
            # absorbs them. There is no yield between this computation and
            # the finish-reason yield, so chunk emission order is unchanged.
            input_tokens, output_tokens, total_tokens, cached_tokens = self._usage_totals(usage_data)

            yield self._make_chunk(finish_reason=finish_reason)

            usage = SimpleNamespace(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            )
            yield SimpleNamespace(
                id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
                choices=[],
                model=self.model,
                usage=usage,
            )
            self._finished = True
        finally:
            watchdog_stop.set()
            if watchdog_thread and watchdog_thread.is_alive():
                watchdog_thread.join(timeout=0.2)
            if self.is_worker:
                if success and not self._interrupted:
                    self.client._update_worker_history(self.messages)
                    if self.worker_lock and self.worker_lock.locked():
                        with contextlib.suppress(Exception):
                            self.worker_lock.release()
                else:
                    self.close()
            else:
                self.close()


def collect_stream_completion(stream: AntigravityStream) -> Any:
    """Consume an AntigravityStream completely and assemble a non-streaming ChatCompletion object.

    The returned object is a :class:`_AwaitableCompletion`: identical in shape
    to the plain ``SimpleNamespace`` this used to return, plus awaitable
    (``await`` yields the very same object) so the async wire can
    ``await client.chat.completions.create(...)`` without a TypeError.
    """
    conversation_id = ""
    model = stream.model
    content_parts: list[str] = []
    tool_calls: list[Any] = []
    finish_reason: str | None = None
    usage: Any = None

    for chunk in stream:
        if not conversation_id and getattr(chunk, "id", None):
            conversation_id = chunk.id
        if getattr(chunk, "model", None):
            model = chunk.model
        if getattr(chunk, "usage", None):
            usage = chunk.usage

        for choice in getattr(chunk, "choices", []):
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta:
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        tool_calls.append(
                            SimpleNamespace(
                                id=getattr(tc, "id", "call_1"),
                                type="function",
                                function=SimpleNamespace(
                                    name=getattr(tc.function, "name", ""),
                                    arguments=getattr(tc.function, "arguments", "{}"),
                                ),
                            )
                        )

    full_content = "".join(content_parts).strip() or None

    # Fallback tool call extraction if tools weren't pre-configured in stream
    if not tool_calls and full_content and "<tool_call>" in full_content:
        extracted_calls, cleaned = _parse_tool_block(full_content)
        if extracted_calls:
            tool_calls = extracted_calls
            full_content = cleaned or None
            finish_reason = "tool_calls"

    message = SimpleNamespace(
        role="assistant",
        content=full_content,
        tool_calls=tool_calls if tool_calls else None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
    )
    if not usage:
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )

    return _AwaitableCompletion(
        id=conversation_id or f"agy-{int(time.time() * 1000)}",
        choices=[choice],
        usage=usage,
        model=model,
    )
