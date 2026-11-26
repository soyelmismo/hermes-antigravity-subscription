"""Preserve active requests and compact oversized historical tool calls."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import prompt as agy_prompt  # noqa: E402


class PendingUserRequestTests(unittest.TestCase):
    def test_short_request_stays_verbatim(self):
        short = "Check why nginx returned 502 and summarize the fix."
        section = agy_prompt._pending_user_request_section(short)
        self.assertIn(f"User:\n{short}\n\n", section)
        self.assertNotIn("already in conversation above", section)

    def test_oversized_request_preserves_the_instruction_in_the_middle(self):
        big = "A" * 1500 + "\nFix the TypeError in report.py.\n" + "Z" * 1500
        section = agy_prompt._pending_user_request_section(big)
        self.assertIn(f"User:\n{big}\n\n", section)
        delta = agy_prompt._format_delta_prompt(
            [{"role": "tool", "tool_call_id": "c_log", "content": "log checked"}],
            pending_user_text=big,
        )
        self.assertIn(big, delta)


class HistoricalToolCallPruningTests(unittest.TestCase):
    def test_prunes_oversized_older_assistant_tool_calls_keeping_recent_8_and_small_old_intact(self):
        msgs: list[dict] = [{"role": "user", "content": "start"}]
        for i in range(9):
            big_body = f"OLD_BODY_{i}_" + ('"漢' * 2000)
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c_old_{i}",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": f"/tmp/old_{i}.py", "content": big_body}),
                            },
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": f"c_old_{i}", "content": "ok"})
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c_old_small",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": json.dumps({"command": "echo KEEP_SMALL_OLD"})},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": "c_old_small", "content": "ok"})

        for i in range(8):
            rec_body = f"RECENT_BODY_{i}_" + ("y" * 2000)
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c_rec_{i}",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": f"/tmp/rec_{i}.py", "content": rec_body}),
                            },
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": f"c_rec_{i}", "content": "ok"})
        msgs.append({"role": "user", "content": "summarize"})

        full = agy_prompt._format_messages_as_prompt(msgs)
        notes = [line for line in full.splitlines() if line.startswith("Historical tool call")]
        self.assertEqual(len(notes), 9)
        for i, note in enumerate(notes):
            self.assertNotIn(f"OLD_BODY_{i}_", full)
            self.assertIn(f'id="c_old_{i}"', note)
            self.assertIn(f'path="/tmp/old_{i}.py"', note)
            self.assertIn(f"Tool Result (c_old_{i}):", full)
            self.assertLessEqual(agy_prompt._prompt_weight(note), 500)
        self.assertIn("KEEP_SMALL_OLD", full)
        for i in range(8):
            self.assertIn(f"RECENT_BODY_{i}_" + ("y" * 2000), full)
        transcript = full[full.index("Conversation transcript:"):]
        parsed_calls, _ = agy_prompt._extract_tagged_tool_calls(transcript)
        self.assertEqual([c.id for c in parsed_calls], ["c_old_small", *[f"c_rec_{i}" for i in range(8)]])

    def test_preserves_extreme_single_recent_tool_call_above_20k(self):
        huge = "HEAD_HUGE_" + ("z" * 25000) + "_TAIL_HUGE"
        msgs = [
            {"role": "user", "content": "write huge"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c_huge",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": json.dumps({"path": "/tmp/huge.txt", "content": huge})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c_huge", "content": "ok"},
            {"role": "user", "content": "done?"},
        ]
        full = agy_prompt._format_messages_as_prompt(msgs)
        self.assertIn("HEAD_HUGE_", full)
        self.assertIn("_TAIL_HUGE", full)
        transcript = full[full.index("Conversation transcript:"):]
        parsed_calls, _ = agy_prompt._extract_tagged_tool_calls(transcript)
        self.assertEqual(len(parsed_calls), 1)
        args_obj = json.loads(parsed_calls[0].function.arguments)
        self.assertEqual(args_obj, {"path": "/tmp/huge.txt", "content": huge})

    def test_note_budget_includes_escaping_and_cannot_create_tool_frames(self):
        close_tag = "</" + "tool_call>"
        path = "/example/" + close_tag
        note = agy_prompt._historical_tool_call_note(
            "c_quoted", "write_file", json.dumps({"path": path, "url": '\\漢' * 300}), 500
        )
        self.assertNotIn(close_tag, note)
        self.assertIn(r'\u003c/tool_call\u003e', note)
        self.assertLessEqual(agy_prompt._prompt_weight(note), 500)
        self.assertEqual(agy_prompt._extract_tagged_tool_calls(note)[0], [])


if __name__ == "__main__":
    unittest.main()
