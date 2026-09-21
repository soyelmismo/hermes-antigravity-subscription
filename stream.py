"""Streaming iterator and response assembly for Antigravity stream-json events."""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator

try:
    from .prompt import _longest_tool_call_prefix_match, _parse_tool_block
except ImportError:
    from prompt import _longest_tool_call_prefix_match, _parse_tool_block

logger = logging.getLogger(__name__)


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
    ):
        self.proc = proc
        self.client = client
        self.model = model
        self.timeout = timeout
        self.has_tools = bool(tools)
        self.is_worker = is_worker
        self.worker_lock = worker_lock
        self.messages = messages
        self.conversation_id = ""
        self._closed = False
        self._interrupted = False
        self._finished = False
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

        try:
            while time.monotonic() < deadline:
                line = self.proc.stdout.readline() if self.proc.stdout else ""
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
                                        if extra_text:
                                            has_content = True
                                            yield self._make_chunk(content=extra_text)
                                    else:
                                        break
                                else:
                                    idx = text_buffer.find("<tool_call")
                                    if idx != -1:
                                        if idx > 0:
                                            safe_text = text_buffer[:idx]
                                            has_content = True
                                            yield self._make_chunk(content=safe_text)
                                        text_buffer = text_buffer[idx:]
                                        in_tool_call = True
                                    else:
                                        k = _longest_tool_call_prefix_match(text_buffer)
                                        if k > 0:
                                            safe_text = text_buffer[:-k]
                                            if safe_text:
                                                has_content = True
                                                yield self._make_chunk(content=safe_text)
                                            text_buffer = text_buffer[-k:]
                                            break
                                        else:
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

            if text_buffer:
                if self.has_tools or "<tool_call>" in text_buffer:
                    parsed_calls, extra_text = _parse_tool_block(text_buffer)
                    if parsed_calls:
                        has_tool_calls = True
                        for call_delta in parsed_calls:
                            yield self._make_chunk(tool_calls=[call_delta])
                    if extra_text:
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

                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")

                if not has_tool_calls and not has_content and returncode != 0:
                    err_detail = error_msg or stderr_out.strip() or f"Process exited with return code {returncode}"
                    raise RuntimeError(f"Antigravity execution failed: {err_detail}")
            else:
                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")

            finish_reason = "tool_calls" if has_tool_calls else "stop"
            yield self._make_chunk(finish_reason=finish_reason)

            input_tokens = int(usage_data.get("input_tokens", 0) or 0)
            output_tokens = int(usage_data.get("output_tokens", 0) or 0)
            total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens) or 0)
            cached_tokens = int(usage_data.get("cache_read_tokens", 0) or 0)

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
