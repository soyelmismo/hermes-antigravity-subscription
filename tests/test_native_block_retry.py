"""A blocked native tool gets one corrective retry, not a fabricated answer."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import _NATIVE_BLOCK_NOTE, AntigravityClient  # noqa: E402
from prompt import _EMPTY_TOOL_RESPONSE_NUDGE  # noqa: E402
from stream import NativeToolProtocolError  # noqa: E402
import test_client_lifecycle as lifecycle  # noqa: E402

MODEL = "gemini-3.8-flash"
TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]
MESSAGES = [
    {"role": "system", "content": "You are Hermes."},
    {"role": "user", "content": "list the files"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
]


def _proc(lines: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = lines + [""]
    return proc


def _blocked(conversation_id: str, tool: str = "read_file", *, prefix: str = "") -> list[str]:
    events: list[dict[str, Any]] = [{"event": "init", "conversation_id": conversation_id}]
    if prefix:
        events.append({"event": "step_update", "step_update": {"text_delta": prefix}})
    events.append({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": tool}})
    return [json.dumps(event) + "\n" for event in events]


def _answered(conversation_id: str, text: str) -> list[str]:
    return lifecycle._worker_turn_lines(conversation_id, text)


def _sent(proc: MagicMock) -> list[str]:
    return [json.loads(c.args[0])["message"]["content"] for c in proc.stdin.write.call_args_list]


class NativeBlockRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        lifecycle._pin_host_dependent_seams(self)

    def _run(self, client: AntigravityClient, turns: list[list[str]], messages, tools=TOOLS):
        """Run one request per entry of ``turns``; returns the agy processes."""
        # Model resolution and the agent lookup spawn `agy models` / `agy
        # --version` through the same Popen: warm them so the mock only sees workers.
        client._resolve_model_and_effort(MODEL, None)
        client._agent_args()
        procs = [_proc(lines) for lines in turns]
        with patch("subprocess.Popen", side_effect=procs):
            results = [
                client.chat.completions.create(model=MODEL, messages=messages, tools=tools, stream=False)
                for _ in turns
            ]
        return procs, results

    def test_the_first_blocked_step_ends_empty_and_the_retry_carries_the_note(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        draft = 'I will check. <tool_call>{"id":"a","function":{"name":"terminal","arguments":"{}"}}</tool_call>'
        _, results = self._run(client, [_blocked("conv-a", prefix=draft)], list(MESSAGES))
        self.assertEqual(results[0].choices[0].message.content, None)
        self.assertFalse(results[0].choices[0].message.tool_calls)
        self.assertEqual(results[0].choices[0].finish_reason, "stop")
        self.assertEqual(client._native_block_note, MESSAGES)

        # Hermes nudges with the same history; its hint edits only the tail.
        retry = [dict(m) for m in MESSAGES]
        retry[-1]["content"] = MESSAGES[-1]["content"] + "\n\n" + _EMPTY_TOOL_RESPONSE_NUDGE
        procs, results = self._run(client, [_answered("conv-b", "a.txt and b.txt")], retry)

        prompt = _sent(procs[0])[0]
        self.assertTrue(prompt.endswith(_NATIVE_BLOCK_NOTE), prompt[-600:])
        example = json.loads(_NATIVE_BLOCK_NOTE.split("<tool_call>")[1].split("</tool_call>")[0])
        self.assertEqual(example["function"]["arguments"], {"argument": "value"})
        self.assertEqual(results[0].choices[0].message.content, "a.txt and b.txt")
        self.assertIsNone(client._native_block_note)

    def test_read_file_then_invalid_terminal_reaches_hermes_as_a_provider_error(self) -> None:
        from agent.error_classifier import FailoverReason, classify_api_error

        client = AntigravityClient()
        self.addCleanup(client.close)

        _, results = self._run(client, [_blocked("conv-a")], list(MESSAGES))
        self.assertIsNone(results[0].choices[0].message.content)
        # v2026.9.24 sends a synthetic assistant/user pair after tool results.
        retry = [*MESSAGES, {"role": "assistant", "content": "(empty)"},
                 {"role": "user", "content": _EMPTY_TOOL_RESPONSE_NUDGE}]
        bad = _proc(_blocked("conv-b", "terminal", prefix="I will try again."))
        with patch("subprocess.Popen", return_value=bad), patch.object(client, "_terminate_process") as terminate:
            stream = client.chat.completions.create(model=MODEL, messages=retry, tools=TOOLS, stream=True)
            with self.assertRaisesRegex(NativeToolProtocolError, "native tool protocol.*terminal") as caught:
                next(stream)
            terminate.assert_called_once_with(bad)
        verdict = classify_api_error(caught.exception, provider="antigravity-subscription-directsdk")
        self.assertEqual(verdict.reason, FailoverReason.incomplete_response)
        self.assertTrue(verdict.should_fallback)
        self.assertFalse(verdict.retryable or verdict.should_compress or verdict.should_rotate_credential)
        self.assertIn("[HOST]", _sent(bad)[0])
        self.assertIsNone(client._native_block_note)
        self.assertFalse(client._worker_lock.locked())

    def test_without_host_tools_the_block_fails_without_retry(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        with self.assertRaisesRegex(NativeToolProtocolError, "read_file"):
            self._run(client, [_blocked("conv-a")], list(MESSAGES), tools=None)
        self.assertIsNone(client._native_block_note)

    def test_an_unrelated_request_drops_the_note(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        self._run(client, [_blocked("conv-a")], list(MESSAGES))
        other = [*MESSAGES[:-1], {**MESSAGES[-1], "content": "different tool result"}]
        procs, _ = self._run(client, [_answered("conv-b", "fine")], other)

        self.assertNotIn("[HOST]", _sent(procs[0])[0])
        self.assertIsNone(client._native_block_note)


if __name__ == "__main__":
    unittest.main()
