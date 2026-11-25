"""A neutralized native tool step is retried, not shown as the answer (issue #61).

Since soyelmismo/hermes-antigravity-subscription#52 a blocked native step
ended the turn with the notice as its final answer. Hermes read that as a
completed text reply: the turn ended and the user saw the notice instead of
the task.

Now the first blocked step in a row ends the turn empty. Hermes' empty-response
ladder re-asks the model, and the plugin appends a host note to that retry's
prompt. A second blocked step in a row keeps the notice, because Hermes fails
a turn over to another provider after two empty replies.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import _NATIVE_BLOCK_NOTE, AntigravityClient  # noqa: E402
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


def _blocked(conversation_id: str, tool: str = "read_file") -> list[str]:
    return [
        json.dumps({"event": "init", "conversation_id": conversation_id}) + "\n",
        json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": tool}}) + "\n",
    ]


def _answered(conversation_id: str, text: str) -> list[str]:
    return lifecycle._worker_turn_lines(conversation_id, text)


def _sent(proc: MagicMock) -> str:
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

        procs, results = self._run(client, [_blocked("conv-a")], list(MESSAGES))
        self.assertEqual(results[0].choices[0].message.content, None)
        self.assertEqual(results[0].choices[0].finish_reason, "stop")
        self.assertEqual(client._native_block_streak, 1)

        # Hermes nudges with the same history; its hint edits only the tail.
        retry = [dict(m) for m in MESSAGES]
        retry[-1]["content"] = MESSAGES[-1]["content"] + "\n\n(continue)"
        procs, results = self._run(client, [_answered("conv-b", "a.txt and b.txt")], retry)

        prompt = _sent(procs[0])[0]
        self.assertTrue(prompt.endswith(_NATIVE_BLOCK_NOTE.format(tool="read_file")), prompt[-300:])
        self.assertEqual(results[0].choices[0].message.content, "a.txt and b.txt")
        # A compliant turn ends the streak and spends the note.
        self.assertEqual(client._native_block_streak, 0)
        self.assertIsNone(client._native_block_note)

    def test_a_repeated_blocked_step_shows_the_notice_again(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        procs, results = self._run(client, [_blocked("conv-a"), _blocked("conv-b", "run_command")], list(MESSAGES))

        self.assertEqual(results[0].choices[0].message.content, None)
        self.assertIn("run_command", results[1].choices[0].message.content)
        self.assertIn("blocked by host security policy", results[1].choices[0].message.content)
        self.assertEqual(client._native_block_streak, 2)

    def test_without_host_tools_the_notice_stays(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        procs, results = self._run(client, [_blocked("conv-a")], list(MESSAGES), tools=None)

        self.assertIn("blocked by host security policy", results[0].choices[0].message.content)
        self.assertEqual(client._native_block_streak, 0)
        self.assertIsNone(client._native_block_note)

    def test_an_unrelated_request_drops_the_note(self) -> None:
        client = AntigravityClient()
        self.addCleanup(client.close)

        self._run(client, [_blocked("conv-a")], list(MESSAGES))
        other = [{"role": "user", "content": "something else"}]
        procs, _ = self._run(client, [_answered("conv-b", "fine")], other)

        self.assertNotIn("[HOST]", _sent(procs[0])[0])
        self.assertEqual(client._native_block_streak, 0)
        self.assertIsNone(client._native_block_note)


if __name__ == "__main__":
    unittest.main()
