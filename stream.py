"""Streaming iterator and response assembly for Antigravity stream-json events."""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import threading
import time
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
            yield self._make_chunk(finish_reason=finish_reason)

            input_tokens, output_tokens, total_tokens, cached_tokens = self._usage_totals(usage_data)

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
    """Consume an AntigravityStream completely and assemble a non-streaming ChatCompletion object."""
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

    return SimpleNamespace(
        id=conversation_id or f"agy-{int(time.time() * 1000)}",
        choices=[choice],
        usage=usage,
        model=model,
    )
