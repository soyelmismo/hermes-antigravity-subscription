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
    # Even a complete call is provisional until the entire turn validates.
    assert calls == []
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
    assert calls == []
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
    assert calls == []
    assert finishes == []


# ---------------------------------------------------------------------------
# Native tool neutralization must not flush buffered frame text or report
# success when the colocated worker cannot make a corrective retry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("partial", ['<tool_call>{"id":"x",', "<tool_cal"])
def test_native_tool_step_neutralization_does_not_flush_partial_block(partial):
    events = [
        {"event": "step_update", "step_update": {"text_delta": partial}},
        {
            "event": "step_update",
            "step_update": {"step_type": "tool", "tool_name": "run_command"},
        },
    ]
    stream, _proc_mock, client = _stream(events)

    with pytest.raises(RuntimeError, match="native tool protocol.*run_command"):
        next(stream)
    client._terminate_process.assert_called()


# ---------------------------------------------------------------------------
# A closed but unparseable frame is invalid protocol, never assistant text
# ---------------------------------------------------------------------------

# One representative per non-protocol / non-tool shape that falls back to plain text:
# absent function, blank function name, non-object function, non-object top level, empty body.
UNPARSEABLE_FRAME_BODIES = [
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
def test_closed_unparseable_frame_falls_back_to_text_in_stream(body):
    raw = f"<tool_call>{body}</tool_call>"
    stream, _proc_mock, _client = _stream(_events(raw))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == raw
    assert finishes == ["stop"]


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_in_result_response_falls_back_to_text(body):
    raw = f"<tool_call>{body}</tool_call>"
    stream, _proc_mock, _client = _stream(_events(final=raw))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == raw
    assert finishes == ["stop"]


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_after_a_complete_call_in_stream(body):
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), f"<tool_call>{body}</tool_call>")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_closed_unparseable_frame_after_a_complete_call_in_result_response():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("one") + "<tool_call>{}</tool_call>")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_tool_call_tag_inside_markdown_code_streams_as_text():
    prose = (
        "Example:\n\n```xml\n"
        "<tool_call>{\"bad\": \"json\"</tool_call>\n"
        "```\n\nDone."
    )
    stream, _proc_mock, _client = _stream(_events(prose))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == prose
    assert finishes == ["stop"]


def test_post_loop_incomplete_tool_call_inside_open_fence_is_plain_text():
    stream, _proc_mock, _client = _stream(
        _events("```xml\n<tool", "_call>{\"unclosed\": true")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "<tool_call>{\"unclosed\": true" in "".join(contents)
    assert finishes == ["stop"]


def test_post_loop_unparseable_tool_call_inside_open_fence_no_warning(caplog):
    stream, _proc_mock, _client = _stream(
        _events("```xml\n", "<tool_call>{}</tool_call>")
    )

    with caplog.at_level("WARNING"):
        calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "<tool_call>{}</tool_call>" in "".join(contents)
    assert finishes == ["stop"]
    assert not any("Unparseable <tool_call>" in r.message for r in caplog.records)


def test_indented_code_block_ignored_and_escaped_backtick_parsed():
    stream, _proc_mock, _client = _stream(
        _events(
            "    <tool_call>{bad}</tool_call>\n\n",
            f"\\`{_block('one')}\\`",
        )
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert "<tool_call>{bad}</tool_call>" in "".join(contents)
    assert finishes == ["tool_calls"]


def test_commonmark_edge_cases_in_fences_and_headings():
    stream, _proc_mock, _client = _stream(
        _events(
            "# `<tool_call>{bad}</tool_call>` #\n\n",
            "```xml\n    ```\n<tool_call>{still_fenced}</tool_call>\n```\n",
            "```xml\n``` extra\n<tool_call>{still_fenced_2}</tool_call>\n```",
        )
    )

    calls, contents, finishes = _consumed(list(stream))
    joined = "".join(contents)
    assert calls == []
    assert "<tool_call>{bad}</tool_call>" in joined
    assert "<tool_call>{still_fenced}</tool_call>" in joined
    assert "<tool_call>{still_fenced_2}</tool_call>" in joined
    assert finishes == ["stop"]


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


def test_prompt_and_history_use_plain_json_object_arguments():
    from prompt import _PROMPT_PREAMBLE, _format_delta_prompt, _format_messages_as_prompt

    preamble = "\n".join(_PROMPT_PREAMBLE)
    assert "'arguments': {...}" in preamble
    assert "JSON object" in preamble

    messages = [
        {"role": "user", "content": "check logs"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": r'{"command": "echo \"hi\""}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "hi"},
    ]
    full_prompt = _format_messages_as_prompt(messages)
    delta_prompt = _format_delta_prompt(messages[1:], pending_user_text="check logs")
    expected_obj = r'"arguments": {"command": "echo \"hi\""}'
    assert expected_obj in full_prompt
    assert expected_obj in delta_prompt
    assert "JSON object for 'arguments'" in delta_prompt


def test_parse_tool_block_accepts_object_arguments_and_repairs_invalid_escapes():
    body = (
        r'{"id": "c1", "type": "function", "function": {'
        r'"name": "search_files", '
        r'"arguments": {"pattern": "foo\s+bar\.py|\'baz\'\n\u0041"}}}'
    )
    calls, cleaned = _parse_tool_block(f"<tool_call>{body}</tool_call>")
    assert len(calls) == 1
    assert calls[0].function.name == "search_files"
    assert json.loads(calls[0].function.arguments) == {
        "pattern": "foo\\s+bar\\.py|'baz'\nA"
    }
    assert cleaned == ""


def test_parse_tool_block_repairs_string_encoded_arguments_with_invalid_escapes():
    body = (
        r'{"id": "c1", "type": "function", "function": {'
        r'"name": "search_files", '
        r'"arguments": "{\"pattern\": \"foo\s+bar\"}"}}'
    )
    calls, cleaned = _parse_tool_block(f"<tool_call>{body}</tool_call>")
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"pattern": r"foo\s+bar"}
    assert cleaned == ""


def test_parse_tool_block_bare_json_fallback_handles_object_arguments():
    bare = r'{"id":"c1","type":"function","function":{"name":"search_files","arguments":{"pattern":"foo\s+bar","nested":{"brace":"}"}}}}'
    calls, cleaned = _parse_tool_block(f"lead {bare} trail")
    assert len(calls) == 1
    assert calls[0].function.name == "search_files"
    assert json.loads(calls[0].function.arguments) == {
        "pattern": r"foo\s+bar",
        "nested": {"brace": "}"},
    }
    assert cleaned == "lead\ntrail"


def test_adjacent_unclosed_parallel_tool_calls_recover_all_calls():
    raw = (
        '<tool_call>{"id":"c1","type":"function","function":{"name":"write_file",'
        '"arguments":{"path":"/tmp/a.txt","content":"first"}}}\n'
        '<tool_call>{"id":"c2","type":"function","function":{"name":"write_file",'
        '"arguments":{"path":"/tmp/b.txt","content":"second"}}}</tool_call>'
    )
    stream, _proc_mock, _client = _stream(_events(raw[:45], raw[45:]))
    calls, contents, finishes = _consumed(list(stream))
    assert [(c.id, c.function.name, json.loads(c.function.arguments)) for c in calls] == [
        ("c1", "write_file", {"path": "/tmp/a.txt", "content": "first"}),
        ("c2", "write_file", {"path": "/tmp/b.txt", "content": "second"}),
    ]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_dense_html_unescaped_quotes_and_odd_quote_close_recover_cleanly():
    # Even quotes: dense HTML attributes inside plain-object arguments.
    html = "".join(f'<td id="c{i}" class="cell" alt="">v{i}</td>' for i in range(60))
    even_raw = (
        '<tool_call>{"id":"html_1","type":"function","function":{"name":"write_file",'
        '"arguments":{"path":"/tmp/page.html","content":"' + html + '"}}}</tool_call>'
    )
    stream, _proc_mock, _client = _stream(_events(even_raw))
    calls, contents, finishes = _consumed(list(stream))
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"path": "/tmp/page.html", "content": html}
    assert contents == []
    assert finishes == ["tool_calls"]

    # Odd unescaped quote: strict _find_tool_call_close sees parity inversion,
    # end-of-turn recovery still closes and repairs the call.
    odd_raw = (
        '<tool_call>{"id":"html_2","type":"function","function":{"name":"write_file",'
        '"arguments":{"path":"/tmp/note.html","content":"<p>He said "hi</p>"}}}</tool_call>'
    )
    stream2, _proc_mock2, _client2 = _stream(_events(odd_raw))
    calls2, contents2, finishes2 = _consumed(list(stream2))
    assert len(calls2) == 1
    assert json.loads(calls2[0].function.arguments) == {
        "path": "/tmp/note.html",
        "content": '<p>He said "hi</p>',
    }
    assert contents2 == []
    assert finishes2 == ["tool_calls"]


def test_broken_tool_call_json_withheld_and_not_leaked_as_prose():
    # Broken outer JSON naming a real tool must never leak raw <tool_call> XML to assistant prose.
    broken = '<tool_call>{"id":"x","function":{"name":"write_file","arguments":{"path":"/tmp/a" "content":"hi"}}}</tool_call>'
    stream, _proc_mock, _client = _stream(_events(broken), is_worker=False)
    (calls, contents, _finishes), err = _consume_until_error(stream)
    assert calls == []
    assert contents == []
    assert "write_file" in err and "not valid JSON" in err


def test_broken_outer_tool_call_triggers_worker_correction_reask():
    broken = '<tool_call>{"id":"x","type":"function","function":{"name":"write_file","arguments":{"path":"/tmp/a" "content":"hi"}}}</tool_call>'
    fixed = '<tool_call>{"id":"x","type":"function","function":{"name":"write_file","arguments":{"path":"/tmp/a","content":"hi"}}}</tool_call>'
    events = _events(broken) + _events(fixed)
    proc = _proc([json.dumps(e) for e in events], poll=None)
    stdin_buf = io.StringIO()
    proc.stdin = stdin_buf
    client = MagicMock()
    client._isolated_gemini_dir = None
    client._cancelled = False
    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        tools=[{"type": "function", "function": {"name": "write_file"}}],
        is_worker=True,
    )
    calls, contents, finishes = _consumed(list(stream))
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"path": "/tmp/a", "content": "hi"}
    assert contents == []
    assert finishes == ["tool_calls"]
    assert "NOT executed" in stdin_buf.getvalue()


def test_literal_quoted_tool_call_tag_in_prose_does_not_trip_parser():
    prose = 'Never emit "<tool_call>" or \'<tool_call>\' directly in plain text.'
    stream, _proc_mock, _client = _stream(_events(prose))
    calls, contents, finishes = _consumed(list(stream))
    assert calls == []
    assert "".join(contents) == prose
    assert finishes == ["stop"]


def test_cross_turn_reused_tool_call_id_is_uniquified():
    prior_messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    proc = _proc([json.dumps(e) for e in _events(_block("second", identifier="call_1"))])
    client = MagicMock()
    client._isolated_gemini_dir = None
    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        tools=[{"type": "function", "function": {"name": "write_file"}}],
        is_worker=False,
        messages=prior_messages,
    )
    calls, contents, finishes = _consumed(list(stream))
    assert len(calls) == 1
    assert calls[0].id != "call_1"
    assert contents == []
    assert finishes == ["tool_calls"]


def test_parse_tool_block_bare_json_fallback_leaves_unparseable_text_untouched():
    raw = '{"id":"c1","type":"function","function":{"name":"search_files","arguments":{not valid}}'
    calls, cleaned = _parse_tool_block(raw)
    assert calls == []
    assert cleaned == raw
