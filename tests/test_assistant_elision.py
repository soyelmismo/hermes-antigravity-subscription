"""Tests for lossless elision of verbatim echoed assistant turns on delta continuations."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import client as agy_client  # noqa: E402


class _SequencedMockWorker:
    """Fake Popen worker recording every user payload written to stdin."""

    def __init__(self, responses: list[str], *, conv_id: str = "conv-tok"):
        self._responses = list(responses)
        self._conv_id = conv_id
        self.written_prompts: list[str] = []
        self.pid = os.getpid()
        self.stderr = io.StringIO("")
        self.stdin = self
        self.stdout = self
        # Like agy, announce the conversation at start-up, before any input:
        # the resume check reads this line before it sends the delta.
        self._buf: list[str] = [
            json.dumps({"event": "init", "conversation_id": conv_id}) + "\n"
        ]
        self._closed = False

    def write(self, data: str) -> int:
        obj = json.loads(data.strip())
        self.written_prompts.append(obj["message"]["content"])
        resp = self._responses.pop(0)
        self._buf = [
            json.dumps({"event": "init", "init": {"conversation_id": self._conv_id}}) + "\n",
            json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": resp}}) + "\n",
            json.dumps({"event": "result", "result": {"conversation_id": self._conv_id, "status": "SUCCESS", "usage": {"input_tokens": 100, "output_tokens": 20}}}) + "\n",
        ]
        return len(data)

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        return self._buf.pop(0) if self._buf else ""

    def poll(self):
        return 0 if self._closed else None

    def terminate(self):
        self._closed = True

    def wait(self, timeout=None):
        self._closed = True
        return 0


class AssistantElisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hermes_agy_elide_")
        self.addCleanup(self.tmp.cleanup)
        self.client = agy_client.AntigravityClient(cwd=self.tmp.name)
        self.addCleanup(self.client.close)

    def test_elides_verbatim_assistant_reply_on_continuation(self):
        secret_reply = "UNIQUE_ASSISTANT_TEXT_98765 " + ("detail " * 40)
        worker = _SequencedMockWorker([secret_reply, "second reply"])
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "first question"}]
            c1 = self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, stream=False)
            self.assertEqual(c1.choices[0].message.content, secret_reply.strip())
            self.assertEqual(self.client._worker_history, m1)

            m2 = m1 + [
                {"role": "assistant", "content": c1.choices[0].message.content},
                {"role": "user", "content": "second question"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, stream=False)

        self.assertEqual(len(worker.written_prompts), 2)
        self.assertNotIn("UNIQUE_ASSISTANT_TEXT_98765", worker.written_prompts[1])
        self.assertIn("second question", worker.written_prompts[1])

    def test_elides_verbatim_assistant_tool_call_on_delta(self):
        tc = {
            "id": "call_exact_1",
            "type": "function",
            "function": {"name": "write_file", "arguments": {"path": "/tmp/a.py", "content": "BIG_FILE_PAYLOAD_XYZ"}},
        }
        worker = _SequencedMockWorker([f"<tool_call>{json.dumps(tc)}</tool_call>", "all done"])
        tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "write a.py"}]
            c1 = self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, tools=tools, stream=False)
            emitted_tc = c1.choices[0].message.tool_calls[0]
            m2 = m1 + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": emitted_tc.id, "type": "function", "function": {"name": emitted_tc.function.name, "arguments": emitted_tc.function.arguments}}
                    ],
                },
                {"role": "tool", "tool_call_id": emitted_tc.id, "name": "write_file", "content": "Wrote ok"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, tools=tools, stream=False)

        self.assertEqual(len(worker.written_prompts), 2)
        self.assertNotIn("BIG_FILE_PAYLOAD_XYZ", worker.written_prompts[1])
        self.assertIn("Tool Result (call_exact_1):\nWrote ok", worker.written_prompts[1])

    def test_retains_assistant_when_mutated_by_host(self):
        worker = _SequencedMockWorker(["original answer", "follow up"])
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "q1"}]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, stream=False)
            m2 = m1 + [
                {"role": "assistant", "content": "EDITED_BY_MIDDLEWARE"},
                {"role": "user", "content": "q2"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, stream=False)

        self.assertIn("EDITED_BY_MIDDLEWARE", worker.written_prompts[1])

    def test_retains_assistant_when_call_id_was_synthesized(self):
        tc_no_id = {"type": "function", "function": {"name": "terminal", "arguments": {"command": "echo hi"}}}
        worker = _SequencedMockWorker([f"<tool_call>{json.dumps(tc_no_id)}</tool_call>", "done"])
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "run"}]
            c1 = self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, tools=tools, stream=False)
            tc = c1.choices[0].message.tool_calls[0]
            self.assertIsNone(self.client._worker_last_assistant)
            m2 = m1 + [
                {"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
                {"role": "tool", "tool_call_id": tc.id, "name": "terminal", "content": "hi"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, tools=tools, stream=False)

        self.assertIn("echo hi", worker.written_prompts[1])

    def test_retains_assistant_when_arguments_were_repaired(self):
        raw = '<tool_call>{"id": "call_r1", "type": "function", "function": {"name": "terminal", "arguments": "{\\"command\\": \\"grep a\\|b f\\"}"}}</tool_call>'
        worker = _SequencedMockWorker([raw, "done"])
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "grep"}]
            c1 = self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, tools=tools, stream=False)
            tc = c1.choices[0].message.tool_calls[0]
            self.assertIsNone(self.client._worker_last_assistant)
            m2 = m1 + [
                {"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
                {"role": "tool", "tool_call_id": tc.id, "name": "terminal", "content": "match"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, tools=tools, stream=False)

        self.assertIn("grep", worker.written_prompts[1])

    def test_preserves_elision_across_conversation_resume_and_clears_on_reset(self):
        w1 = _SequencedMockWorker(["RESUMED_ANSWER_555"])
        w2 = _SequencedMockWorker(["after resume"])
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", side_effect=[w1, w2]), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "q1"}]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, stream=False)
            self.client._terminate_worker(keep_conversation=True)
            self.assertIsNone(self.client._worker_last_assistant)
            self.assertIsNotNone(self.client._conversation_last_assistant)

            m2 = m1 + [{"role": "assistant", "content": "RESUMED_ANSWER_555"}, {"role": "user", "content": "q2"}]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, stream=False)
            self.assertNotIn("RESUMED_ANSWER_555", w2.written_prompts[0])

            self.client._terminate_worker(keep_conversation=False)
            self.assertIsNone(self.client._conversation_last_assistant)

    def test_elides_tool_call_when_host_canonicalizes_arguments(self):
        tc = {
            "id": "call_canon_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": {"path": "/tmp/a.py", "content": "CANON_PAYLOAD_XYZ"},
            },
        }
        worker = _SequencedMockWorker([f"<tool_call>{json.dumps(tc)}</tool_call>", "all done"])
        tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]
        with patch("client.is_authenticated", return_value=True), \
                patch("client.subprocess.Popen", return_value=worker), \
                patch("client.custom_agent_args", return_value=[]):
            m1 = [{"role": "user", "content": "write a.py"}]
            c1 = self.client.chat.completions.create(model="gemini-3.8-flash", messages=m1, tools=tools, stream=False)
            emitted = c1.choices[0].message.tool_calls[0]
            # Hermes rewrites every assistant tool-call arguments string on the
            # send path to compact separators and sorted keys before any provider
            # sees it, so the echoed copy never matches the emitted bytes.
            canonical = json.dumps(
                json.loads(emitted.function.arguments), separators=(",", ":"), sort_keys=True
            )
            m2 = m1 + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": emitted.id,
                            "type": "function",
                            "function": {"name": emitted.function.name, "arguments": canonical},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": emitted.id, "name": "write_file", "content": "Wrote ok"},
            ]
            self.client.chat.completions.create(model="gemini-3.8-flash", messages=m2, tools=tools, stream=False)

        self.assertEqual(len(worker.written_prompts), 2)
        self.assertNotIn("CANON_PAYLOAD_XYZ", worker.written_prompts[1])
        self.assertIn("Tool Result (call_canon_1):\nWrote ok", worker.written_prompts[1])


if __name__ == "__main__":
    unittest.main()
