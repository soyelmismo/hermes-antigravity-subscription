"""Regression tests for the <tool_call> frame, flush, and quota watchdog.

Synthetic stream-json only: no agy process, no network, no host state.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stream import AntigravityStream

TOOL_CALL_CLOSE = "</tool_call>"


def _block(content: str, identifier: str = "same", name: str = "write_file") -> str:
    call = {
        "id": identifier,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"path": "fixture.txt", "content": content}),
        },
    }
    return f"<tool_call>{json.dumps(call)}</tool_call>"


def _events(*deltas: str, final: str = "", status: str = "SUCCESS") -> list[dict]:
    events = [
        {"event": "step_update", "step_update": {"text_delta": delta}}
        for delta in deltas
    ]
    events.append({"event": "result", "result": {"status": status, "response": final}})
    return events


def _proc(event_lines: list[str], *, poll: int | None = 0):
    proc = MagicMock()
    proc.pid = None
    proc.stdout = io.StringIO("".join(line + "\n" for line in event_lines))
    proc.stderr = io.StringIO("")
    proc._stderr_drainer = None
    proc.poll.return_value = poll
    proc.wait.return_value = 0
    return proc


def _stream(events: list[dict], *, tools: bool = True, is_worker: bool = False):
    proc = _proc([json.dumps(event) for event in events])
    client = MagicMock()
    client._isolated_gemini_dir = None
    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        tools=[{"type": "function", "function": {"name": "write_file"}}] if tools else None,
        is_worker=is_worker,
    )
    return stream, proc, client


def _consumed(chunks):
    calls = [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in (choice.delta.tool_calls or [])
    ]
    contents = [
        choice.delta.content
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta and choice.delta.content
    ]
    finishes = [
        choice.finish_reason
        for chunk in chunks
        for choice in chunk.choices
        if choice.finish_reason
    ]
    return calls, contents, finishes


def test_closing_tag_inside_json_string_does_not_split_the_call():
    block = _block("text </tool_call> text")
    # Split inside the literal sequence so the streaming loop must carry the
    # candidate close across deltas without treating it as a frame boundary.
    split_at = block.index(TOOL_CALL_CLOSE) + len(TOOL_CALL_CLOSE) - 2
    stream, _proc_mock, _client = _stream(
        _events(block[:split_at], block[split_at:])
    )

    chunks = list(stream)
    calls, contents, finishes = _consumed(chunks)

    assert len(calls) == 1
    assert calls[0].function.name == "write_file"
    assert json.loads(calls[0].function.arguments)["content"] == "text </tool_call> text"
    assert contents == []
    assert finishes == ["tool_calls"]


def test_incomplete_tool_block_fails_instead_of_becoming_assistant_text():
    stream, _proc_mock, _client = _stream(_events('<tool_call>{"id":"incomplete",'))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    _calls, contents, finishes = _consumed(chunks)
    assert contents == []
    assert finishes == []


def test_incomplete_second_tool_block_after_a_complete_call_fails():
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), '<tool_call>{"id":"second",')
    )

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, _contents, finishes = _consumed(chunks)
    # The first, complete call was already emitted and cannot be retracted.
    assert [call.id for call in calls] == ["same"]
    assert finishes == []


def test_tool_call_indices_and_ids_are_unique_across_blocks():
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), _block("two"), _block("three"))
    )

    chunks = list(stream)
    calls, _contents, finishes = _consumed(chunks)

    assert [call.index for call in calls] == [0, 1, 2]
    assert [call.id for call in calls] == ["same", "call_2", "call_3"]
    assert [json.loads(call.function.arguments)["content"] for call in calls] == [
        "one",
        "two",
        "three",
    ]
    assert finishes == ["tool_calls"]


def _init_then_slow_result(seconds: float):
    init = json.dumps({"event": "init", "conversation_id": "fixture"})
    result = json.dumps(
        {"event": "result", "result": {"status": "SUCCESS", "response": "done"}}
    )
    calls = 0

    def readline() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return init
        if calls == 2:
            time.sleep(seconds)
            return result
        return ""

    proc = MagicMock()
    proc.pid = None
    proc.stdout.readline.side_effect = readline
    proc.stderr = io.StringIO("")
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    return proc


def test_watchdog_keeps_probing_after_init_and_uses_wallclock_mtime():
    proc = _init_then_slow_result(1.8)
    client = MagicMock()
    client._isolated_gemini_dir = "/fixture/gemini"
    probe_calls: list[float | None] = []

    def probe(_gemini_dir, min_mtime=None):
        probe_calls.append(min_mtime)
        return None

    before = time.time()
    with patch("stream._check_early_quota_error", side_effect=probe):
        chunks = list(
            AntigravityStream(
                proc=proc,
                client=client,
                model="fixture-only",
                timeout=5.0,
            )
        )
    after = time.time()

    assert probe_calls, "init must not disarm the quota watchdog before content"
    # File mtimes are epoch seconds; a monotonic clock here would be orders
    # of magnitude too small and would never filter a stale log.
    for min_mtime in probe_calls:
        assert min_mtime is not None
        assert before - 2.0 <= min_mtime <= after + 2.0
    assert any(choice.delta.content == "done" for chunk in chunks for choice in chunk.choices)


def test_init_then_quota_before_content_aborts_the_turn(tmp_path: Path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "cli-fixture.log").write_text(
        "RESOURCE_EXHAUSTED: Individual quota reached, code 429\n",
        encoding="utf-8",
    )
    proc = _init_then_slow_result(1.8)
    client = MagicMock()
    client._isolated_gemini_dir = tmp_path
    client._terminate_process = MagicMock()

    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
    )
    with pytest.raises(RuntimeError) as ctx:
        list(stream)

    assert "RESOURCE_EXHAUSTED" in str(ctx.value)
    assert "Individual quota reached" in str(ctx.value)
    client._terminate_process.assert_called()


@pytest.mark.parametrize("is_worker", [False, True])
def test_nonzero_exit_with_stale_quota_log_reports_exit_code(
    tmp_path: Path, is_worker: bool
):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    stale_log = log_dir / "cli-stale.log"
    stale_log.write_text(
        "RESOURCE_EXHAUSTED: Individual quota reached, code 429\n",
        encoding="utf-8",
    )
    old = time.time() - 86400
    os.utime(stale_log, (old, old))

    proc = _proc([], poll=1)
    client = MagicMock()
    client._isolated_gemini_dir = tmp_path
    client._terminate_process = MagicMock()

    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        is_worker=is_worker,
    )
    with pytest.raises(RuntimeError) as ctx:
        list(stream)

    message = str(ctx.value)
    assert "Antigravity execution failed" in message
    assert "return code 1" in message
    assert "RESOURCE_EXHAUSTED" not in message
