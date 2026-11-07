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

from prompt import _has_unfinished_tool_call_frame, _parse_tool_block
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


# ---------------------------------------------------------------------------
# result.response uses the same frame parser as streamed deltas
# ---------------------------------------------------------------------------


def test_complete_tool_call_only_in_result_response_is_parsed():
    stream, _proc_mock, _client = _stream(_events(final=_block("valid")))

    calls, contents, finishes = _consumed(list(stream))

    assert [call.function.name for call in calls] == ["write_file"]
    assert json.loads(calls[0].function.arguments)["content"] == "valid"
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_call_with_literal_closing_tag_in_arguments_is_parsed():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("text </tool_call> text"))
    )

    calls, contents, finishes = _consumed(list(stream))

    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments)["content"] == (
        "text </tool_call> text"
    )
    assert contents == []
    assert finishes == ["tool_calls"]


def test_two_complete_tool_calls_only_in_result_response_are_unique():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("one") + _block("two"))
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.index for call in calls] == [0, 1]
    assert [call.id for call in calls] == ["same", "call_2"]
    assert [json.loads(call.function.arguments)["content"] for call in calls] == [
        "one",
        "two",
    ]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_plain_text_with_tools_stays_content():
    stream, _proc_mock, _client = _stream(_events(final="all done"))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert contents == ["all done"]
    assert finishes == ["stop"]


@pytest.mark.parametrize(
    "final, expected",
    [(_block("valid"), 1), (_block("one") + _block("two"), 2)],
)
def test_result_only_tool_calls_still_parse_without_tools_configured(final, expected):
    stream, _proc_mock, _client = _stream(_events(final=final), tools=False)

    calls, contents, finishes = _consumed(list(stream))

    assert len(calls) == expected
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_prose_mentioning_tag_stays_text_without_tools():
    response = "Use <tool_call> to call tools."
    stream, _proc_mock, _client = _stream(_events(final=response), tools=False)

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert contents == [response]
    assert finishes == ["stop"]


@pytest.mark.parametrize(
    "obj",
    [
        {},
        {"function": {"name": "", "arguments": "{}"}},
        {"function": {"name": "   ", "arguments": "{}"}},
        {"function": {"name": 123, "arguments": "{}"}},
        {"function": "not-a-dict"},
    ],
)
def test_tool_objects_without_a_nonblank_function_name_are_rejected(obj):
    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert calls == []


def test_tool_call_function_name_is_stripped():
    body = '{"function": {"name": "  write_file  ", "arguments": "{}"}}'

    calls, _text = _parse_tool_block(f"<tool_call>{body}</tool_call>")

    assert [call.function.name for call in calls] == ["write_file"]


@pytest.mark.parametrize("identifier", [None, 123, "", "   "])
def test_tool_call_blank_or_non_string_id_falls_back(identifier):
    obj = {"id": identifier, "function": {"name": "write_file", "arguments": "{}"}}

    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert [call.id for call in calls] == ["call_1"]


def test_tool_call_string_id_is_preserved():
    obj = {"id": "call_x", "function": {"name": "write_file", "arguments": "{}"}}

    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert [call.id for call in calls] == ["call_x"]


# ---------------------------------------------------------------------------
# A stream cut inside any part of a frame fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tail", ["<tool_cal", "<tool_call"])
def test_stream_cut_inside_opening_tag_after_a_complete_call_fails(tail):
    stream, _proc_mock, _client = _stream(_events(_block("one"), tail))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, contents, finishes = _consumed(chunks)
    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == []


@pytest.mark.parametrize("tail", ["<tool_cal", "<tool_call"])
def test_stream_cut_inside_opening_tag_without_a_call_fails(tail):
    stream, _proc_mock, _client = _stream(_events(tail))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, contents, finishes = _consumed(chunks)
    assert calls == []
    assert contents == []
    assert finishes == []


def test_stream_cut_inside_closing_tag_after_a_complete_call_fails():
    partial = (
        '<tool_call>{"id":"b","function":{"name":"write_file",'
        '"arguments":"{}"}</tool_ca'
    )
    stream, _proc_mock, _client = _stream(_events(_block("one"), partial))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, _contents, finishes = _consumed(chunks)
    assert [call.id for call in calls] == ["same"]
    assert finishes == []


# ---------------------------------------------------------------------------
# Native tool neutralization keeps its pinned empty-stop behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("partial", ['<tool_call>{"id":"x",', "<tool_cal"])
def test_native_tool_step_neutralization_with_partial_block_does_not_fail(partial):
    events = [
        {"event": "step_update", "step_update": {"text_delta": partial}},
        {
            "event": "step_update",
            "step_update": {"step_type": "tool", "tool_name": "run_command"},
        },
    ]
    stream, _proc_mock, client = _stream(events)

    chunks = list(stream)

    calls, contents, finishes = _consumed(chunks)
    assert calls == []
    assert contents == []
    assert finishes == ["stop"]
    client._terminate_process.assert_called()


# ---------------------------------------------------------------------------
# A closed but unparseable frame is invalid protocol, never assistant text
# ---------------------------------------------------------------------------

# One representative per rejected shape: broken JSON, absent function, blank
# function name, non-object function, non-object top level, empty body.
UNPARSEABLE_FRAME_BODIES = [
    '{"id":"x","function":{"name":"write_file","arguments":"{"}',
    '{"id":"x"}',
    '{"function":{"name":"   ","arguments":"{}"}}',
    '{"function":[]}',
    "[]",
    "",
]


def _consume_until_error(stream):
    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)
    return _consumed(chunks), str(ctx.value)


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_fails_in_stream(body):
    stream, _proc_mock, _client = _stream(_events(f"<tool_call>{body}</tool_call>"))

    (calls, contents, finishes), message = _consume_until_error(stream)

    assert "invalid tool call block" in message
    assert calls == []
    assert contents == []
    assert finishes == []


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_in_result_response_fails(body):
    stream, _proc_mock, _client = _stream(
        _events(final=f"<tool_call>{body}</tool_call>")
    )

    (calls, contents, finishes), message = _consume_until_error(stream)

    assert "invalid tool call block" in message
    assert calls == []
    assert contents == []
    assert finishes == []


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_after_a_complete_call_fails_in_stream(body):
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), f"<tool_call>{body}</tool_call>")
    )

    (calls, contents, finishes), message = _consume_until_error(stream)

    assert "invalid tool call block" in message
    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == []


def test_closed_unparseable_frame_after_a_complete_call_in_result_response_fails():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("one") + "<tool_call>{}</tool_call>")
    )

    (calls, contents, finishes), message = _consume_until_error(stream)

    assert "invalid tool call block" in message
    # The result-only buffer is validated before anything is emitted.
    assert calls == []
    assert contents == []
    assert finishes == []


# ---------------------------------------------------------------------------
# A lone "<" ending prose is text, a longer tag prefix remains a frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, unfinished",
    [
        ("Compare a <", False),
        ("Compare a <t", True),
        ("Compare a <tool_cal", True),
        ("Compare a <tool_call", True),
        ('<tool_call>{"id":"x",', True),
    ],
)
def test_lone_angle_bracket_is_text_but_longer_prefixes_are_frames(text, unfinished):
    assert _has_unfinished_tool_call_frame(text) is unfinished


@pytest.mark.parametrize("tools", [True, False])
def test_prose_ending_with_lone_angle_bracket_stays_content(tools):
    stream, _proc_mock, _client = _stream(_events("Compare a <"), tools=tools)

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == "Compare a <"
    assert finishes == ["stop"]


def test_prose_ending_with_lone_angle_bracket_after_a_complete_call_succeeds():
    stream, _proc_mock, _client = _stream(_events(_block("one"), "Compare a <"))

    calls, _contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert finishes == ["tool_calls"]
