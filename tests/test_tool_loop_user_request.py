"""The user request a tool loop serves must stay in every tool-result prompt.

Regression for long tool loops on the persistent worker: the delta prompt for a
tool-result turn used to end with a generic "continue" line only, so after
several tool calls the model lost the user's question and re-sent an earlier
report from the transcript instead of answering it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from prompt import (  # noqa: E402
    _format_delta_prompt,
    _format_messages_as_prompt,
    _latest_user_text,
)

QUESTION = "Is there any historical PO ledger data at all?"


def _history_with_tool_loop() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hide the menu and deploy."},
        {"role": "assistant", "content": "### Completion report\nMenu hidden and deployed."},
        {"role": "user", "content": QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "supply.db 108K"},
    ]


class TestToolLoopKeepsUserRequest(unittest.TestCase):
    def test_latest_user_text_picks_last_user_message(self):
        self.assertEqual(_latest_user_text(_history_with_tool_loop()), QUESTION)
        self.assertEqual(_latest_user_text([{"role": "assistant", "content": "x"}]), "")

    def test_delta_tool_prompt_restates_pending_request_after_results(self):
        history = _history_with_tool_loop()
        delta = history[-2:]  # assistant tool_call + tool result: no user message
        prompt = _format_delta_prompt(delta, pending_user_text=_latest_user_text(history))

        self.assertIn(QUESTION, prompt)
        # The request is the newest text the model reads, after the tool output.
        self.assertGreater(prompt.rindex(QUESTION), prompt.index("supply.db 108K"))
        self.assertIn("Do NOT repeat earlier reports", prompt)

    def test_delta_tool_prompt_without_pending_text_is_backward_compatible(self):
        delta = _history_with_tool_loop()[-2:]
        prompt = _format_delta_prompt(delta)
        self.assertNotIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
        self.assertIn("Continue the conversation from the latest tool result.", prompt)

    def test_full_tool_prompt_restates_pending_request_at_tail(self):
        prompt = _format_messages_as_prompt(_history_with_tool_loop())
        tail = prompt[prompt.index("### LATEST TOOL RESULTS RECEIVED."):]
        self.assertIn(QUESTION, tail)

    def test_user_turn_prompt_is_unchanged(self):
        history = _history_with_tool_loop()[:4]
        prompt = _format_delta_prompt(history[-1:], pending_user_text=QUESTION)
        self.assertNotIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
        self.assertEqual(prompt.count(QUESTION), 1)


if __name__ == "__main__":
    unittest.main()
