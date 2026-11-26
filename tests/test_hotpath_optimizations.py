"""Tests for lossless hot-path optimizations in prompt.py, stream.py, and client.py."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import client as agy_client  # noqa: E402
import prompt as agy_prompt  # noqa: E402
import stream as agy_stream  # noqa: E402


def _reference_find_tool_call_close(text: str, start: int = 0) -> int:
    """Original character-by-character scanner kept as a reference oracle."""
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if text.startswith(agy_prompt._TOOL_CALL_CLOSE, index):
            return index
    return -1


class FindToolCallCloseParityTests(unittest.TestCase):
    def test_matches_reference_scanner_across_edge_cases(self):
        cases = [
            "",
            "no closing tag here",
            "</tool_call>",
            '{"a": "plain"}</tool_call>',
            '{"a": "</tool_call>"}',
            '{"a": "</tool_call>"}</tool_call>',
            '{"a": "\\"</tool_call>\\""}</tool_call>',
            '{"a": "\\\\"}</tool_call>',
            '{"a": "\\\\\\"</tool_call>"}</tool_call>',
            '{"a": ""}</tool_call>',
            '{"a": "unterminated \\',
            '{"a": "unterminated </tool_call>',
            'prefix {"x": "</tool_call>"} </tool_call> {"y": 1}</tool_call>',
        ]
        for text in cases:
            for start in (0, min(3, len(text)), min(10, len(text))):
                with self.subTest(text=text[:40], start=start):
                    self.assertEqual(
                        agy_prompt._find_tool_call_close(text, start),
                        _reference_find_tool_call_close(text, start),
                    )


class PromptWeightParityTests(unittest.TestCase):
    def test_matches_character_sum_formula(self):
        samples = [
            "",
            "pure ascii 12345 !@#",
            "café résumé naïve",
            "暗号abc日本語",
            "emoji 🚀🔥🧪 + ascii + 中文 + \u007f\u0080\uffff",
            ("ascii line\n" * 200) + "→" + ("more ascii\n" * 200),
        ]
        for text in samples:
            expected = len(text) + sum(1 for ch in text if ord(ch) > 127)
            with self.subTest(sample=text[:30]):
                self.assertEqual(agy_prompt._prompt_weight(text), expected)


class CleanRenderedTextParityTests(unittest.TestCase):
    def test_fast_path_and_regex_path_match(self):
        self.assertEqual(agy_prompt._clean_rendered_text(""), "")
        self.assertEqual(agy_prompt._clean_rendered_text("  plain text  "), "plain text")
        self.assertEqual(
            agy_prompt._clean_rendered_text("hello [screenshot] world"),
            "hello world",
        )
        self.assertEqual(
            agy_prompt._clean_rendered_text("hello [screenshot: top-bar] world"),
            "hello world",
        )
        self.assertEqual(
            agy_prompt._clean_rendered_text("[Image attached at: /tmp/a.png]"),
            "[Image attached at: /tmp/a.png]",
        )


class MarkdownCodeStateTests(unittest.TestCase):
    def test_closed_fence_with_blank_line_does_not_leak_open_fence(self):
        state = agy_prompt._MarkdownCodeState()
        state.advance("```python\nline1\n\nline2\n```\n")
        self.assertFalse(state.in_code())
        # A real <tool_call> right after the closed fence must be detected.
        self.assertEqual(
            agy_prompt._find_unquoted_tool_call_open("<tool_call>{}</tool_call>", state),
            0,
        )

    def test_deferred_compaction_keeps_open_fence_bounded(self):
        state = agy_prompt._MarkdownCodeState()
        state.advance("```python\n")
        for i in range(400):
            state.advance(f"x_{i} = {i} # padding line inside fenced code block\n")
        self.assertTrue(state.in_code())
        self.assertLess(len(state._tail), 200)
        state.advance("```\n")
        self.assertFalse(state.in_code())

    def test_deferred_compaction_keeps_consecutive_closed_fences_bounded(self):
        for separator in ("", "\n", "\nIntro\n\n"):
            with self.subTest(separator=separator):
                state = agy_prompt._MarkdownCodeState()
                for _ in range(400):
                    state.advance("```python\nfirst = 1\n\nsecond = 2\n```\n" + separator)
                self.assertFalse(state.in_code())
                self.assertLess(len(state._tail), 200)

    def test_clone_isolates_pending_chunks(self):
        state = agy_prompt._MarkdownCodeState()
        state.advance("intro paragraph ")
        cloned = state.clone()
        cloned.advance("\n```python\n")
        self.assertTrue(cloned.in_code())
        self.assertFalse(state.in_code())


class StreamHotpathTests(unittest.TestCase):
    @staticmethod
    def _make_ndjson(deltas: list[str], conv_id: str = "conv-hot") -> str:
        lines = [json.dumps({"event": "init", "init": {"conversation_id": conv_id}}) + "\n"]
        for d in deltas:
            lines.append(
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {"step_type": "agent_response", "text_delta": d},
                    }
                )
                + "\n"
            )
        lines.append(
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": conv_id,
                        "status": "SUCCESS",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                }
            )
            + "\n"
        )
        return "".join(lines)

    @staticmethod
    def _dummy_client(proc: Any) -> SimpleNamespace:
        return SimpleNamespace(
            _cancelled=False,
            _isolated_gemini_dir=None,
            _lock=threading.RLock(),
            _worker_proc=proc,
            _active_processes=set(),
            _terminate_process=lambda p: None,
            _terminate_worker=lambda keep_conversation=False: None,
            _update_worker_history=lambda msgs: None,
            _set_worker_conversation_id=lambda cid: None,
            _note_worker_activity=MagicMock(),
            _custom_agent_named=False,
        )

    def test_split_closing_tag_inside_string_and_at_end(self):
        payload1 = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": {"path": "/tmp/a.txt", "content": "literal </tool_call> here"},
            },
        }
        payload2 = {
            "id": "call_2",
            "type": "function",
            "function": {"name": "terminal", "arguments": {"command": "echo ok"}},
        }
        # Split both the literal inner </tool_call> and the real closing </tool_call> across deltas.
        raw1 = "<tool_call>" + json.dumps(payload1) + "</tool_call>"
        raw2 = "\n<tool_call>" + json.dumps(payload2) + "</tool_call>"
        full = raw1 + raw2
        deltas = [full[i : i + 5] for i in range(0, len(full), 5)]

        proc = SimpleNamespace(
            pid=os.getpid(),
            stdout=io.StringIO(self._make_ndjson(deltas)),
            stdin=io.StringIO(),
            stderr=io.StringIO(""),
            poll=lambda: None,
            wait=lambda timeout=None: 0,
        )
        cli = self._dummy_client(proc)
        wlock = threading.Lock()
        wlock.acquire()
        stream = agy_stream.AntigravityStream(
            proc=proc,
            client=cli,
            model="gemini-3.8-flash-low",
            timeout=30.0,
            tools=[
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
            ],
            is_worker=True,
            worker_lock=wlock,
            worker_lock_held=True,
            messages=[{"role": "user", "content": "run"}],
            usage_baseline={},
        )
        calls = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                calls.extend(chunk.choices[0].delta.tool_calls)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].function.name, "write_file")
        self.assertEqual(json.loads(calls[0].function.arguments)["content"], "literal </tool_call> here")
        self.assertEqual(calls[1].function.name, "terminal")
        self.assertEqual(json.loads(calls[1].function.arguments)["command"], "echo ok")

    def test_precomputed_media_paths_skip_duplicate_scan(self):
        proc = SimpleNamespace(stdout=io.StringIO(""), stderr=io.StringIO(""), poll=lambda: 0)
        cli = self._dummy_client(proc)
        with patch.object(agy_stream, "media_reference_paths") as mock_scan:
            st = agy_stream.AntigravityStream(
                proc=proc,
                client=cli,
                model="gemini-3.8-flash-low",
                timeout=10.0,
                messages=[{"role": "user", "content": "[Image](/tmp/x.png)"}],
                media_paths={"/tmp/x.png"},
            )
            mock_scan.assert_not_called()
            self.assertEqual(st._media_paths, {"/tmp/x.png"})

    def test_worker_activity_throttled_within_one_second(self):
        deltas = [f"tok{i} " for i in range(50)]
        proc = SimpleNamespace(
            pid=os.getpid(),
            stdout=io.StringIO(self._make_ndjson(deltas)),
            stdin=io.StringIO(),
            stderr=io.StringIO(""),
            poll=lambda: None,
            wait=lambda timeout=None: 0,
        )
        cli = self._dummy_client(proc)
        wlock = threading.Lock()
        wlock.acquire()
        st = agy_stream.AntigravityStream(
            proc=proc,
            client=cli,
            model="gemini-3.8-flash-low",
            timeout=30.0,
            is_worker=True,
            worker_lock=wlock,
            worker_lock_held=True,
            messages=[{"role": "user", "content": "hi"}],
            usage_baseline={},
        )
        list(st)
        # 52 stdout lines processed in a few milliseconds should note activity once, not 52 times.
        self.assertEqual(cli._note_worker_activity.call_count, 1)


class PriorMessageMediaSinkTests(unittest.TestCase):
    def test_full_prompt_registers_prior_data_url_in_media_sink(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first image"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "follow-up question"},
        ]
        with tempfile.TemporaryDirectory() as tmp_home:
            sink: set[str] = set()
            with patch.object(agy_prompt, "get_hermes_home", return_value=tmp_home):
                agy_prompt._format_messages_as_prompt(messages, media_sink=sink)
            self.assertEqual(len(sink), 1)
            materialized = next(iter(sink))
            self.assertTrue(os.path.isfile(materialized))


if __name__ == "__main__":
    unittest.main()
