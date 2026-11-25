"""Tests for bounded tool-loop user reminders and historical assistant tool-call pruning."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import prompt as agy_prompt  # noqa: E402


class BoundedPendingUserRequestTests(unittest.TestCase):
    def test_short_request_stays_verbatim(self):
        short = "Check why nginx returned 502 and summarize the fix."
        section = agy_prompt._pending_user_request_section(short)
        self.assertIn(f"User:\n{short}\n\n", section)
        self.assertNotIn("already in conversation above", section)

    def test_exact_boundary_stays_verbatim(self):
        boundary = "A" * agy_prompt._PENDING_USER_REQUEST_MAX_CHARS
        section = agy_prompt._pending_user_request_section(boundary)
        self.assertIn(boundary, section)
        self.assertNotIn("already in conversation above", section)

    def test_oversized_request_preserves_head_and_tail_only(self):
        big = "TOP_SPEC_ANCHOR " + ("M" * 4000) + " BOTTOM_QUESTION_ANCHOR"
        section = agy_prompt._pending_user_request_section(big)
        self.assertIn("TOP_SPEC_ANCHOR", section)
        self.assertIn("BOTTOM_QUESTION_ANCHOR", section)
        self.assertIn(f"full user request ({len(big)} chars) already in conversation above", section)
        self.assertLess(len(section), 1400)


class HistoricalToolCallPruningTests(unittest.TestCase):
    def test_prunes_oversized_older_assistant_tool_calls_keeping_recent_8_and_small_old_intact(self):
        msgs: list[dict] = [{"role": "user", "content": "start"}]
        for i in range(9):
            big_body = f"OLD_BODY_{i}_" + ("x" * 2000)
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
        for i in range(9):
            self.assertNotIn("x" * 500, full)
            self.assertIn(f"/tmp/old_{i}.py", full)
        self.assertIn('\\"_truncated\\": true', full)
        self.assertIn("KEEP_SMALL_OLD", full)
        for i in range(8):
            self.assertIn(f"RECENT_BODY_{i}_" + ("y" * 2000), full)
        transcript = full[full.index("Conversation transcript:"):]
        parsed_calls, _ = agy_prompt._extract_tagged_tool_calls(transcript)
        self.assertEqual(len(parsed_calls), 18)
        for c in parsed_calls[:9]:
            args_obj = json.loads(c.function.arguments)
            self.assertTrue(args_obj["_truncated"])
            self.assertIn("path", args_obj)

    def test_caps_extreme_single_recent_tool_call_above_20k(self):
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
        self.assertTrue(args_obj["_truncated"])
        self.assertLess(len(full), 22000)


if __name__ == "__main__":
    unittest.main()
