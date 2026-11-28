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

    def test_full_tool_prompt_anchors_pending_request_above_transcript(self):
        prompt = _format_messages_as_prompt(_history_with_tool_loop())
        head = prompt[: prompt.index("Conversation transcript:")]
        self.assertIn("### ACTIVE OBJECTIVE:", head)
        self.assertIn(QUESTION, head)
        self.assertNotIn("Hide the menu and deploy.", head)

    def test_full_user_prompt_keeps_single_objective(self):
        history = _history_with_tool_loop()[:4]
        prompt = _format_messages_as_prompt(history)
        self.assertEqual(prompt.count("### ACTIVE OBJECTIVE:"), 1)
        self.assertIn("The user has submitted a new request", prompt)

    def test_user_turn_prompt_is_unchanged(self):
        history = _history_with_tool_loop()[:4]
        prompt = _format_delta_prompt(history[-1:], pending_user_text=QUESTION)
        self.assertNotIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
        self.assertEqual(prompt.count(QUESTION), 1)

    def test_compaction_summary_is_not_restated(self):
        padding = "x" * 20_000
        summary = (
            "[CONTEXT COMPACTION — REFERENCE ONLY] " + padding
            + "\n## User Messages (verbatim, newest first)\n> Fix the plugin\n"
            + "\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n\n"
            + "[STILL IN PROGRESS — this is the active request, restated after the "
            + "compaction boundary because it was not finished yet. Continue it; do not start over.]\n"
            + "Fix the plugin"
        )
        history = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "patched"},
        ]
        self.assertEqual(_latest_user_text(history), "Fix the plugin")
        # Tool-result delta restatement strips the 20 KB summary boilerplate.
        prompt = _format_delta_prompt(history[-2:], pending_user_text=_latest_user_text(history))
        self.assertNotIn("CONTEXT COMPACTION", prompt)
        self.assertNotIn(padding[:40], prompt)
        self.assertIn("Fix the plugin", prompt)
        # Recovery-turn delta restatement also strips the summary boilerplate.
        recovery_msg = {
            "role": "user",
            "content": "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.",
        }
        recovery_prompt = _format_delta_prompt([recovery_msg], pending_user_text=summary)
        self.assertNotIn(padding[:40], recovery_prompt)
        self.assertIn("Fix the plugin", recovery_prompt)
        # Full prompt ending on a compaction summary in multi-message history:
        # summary lives once in transcript; ACTIVE OBJECTIVE & tail restate only the live ask.
        multi = [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": summary},
        ]
        full_multi = _format_messages_as_prompt(multi)
        head = full_multi[: full_multi.index("Conversation transcript:")]
        tail = full_multi[full_multi.index("### LATEST USER REQUEST"):]
        self.assertNotIn(padding[:40], head)
        self.assertIn("Fix the plugin", head)
        self.assertNotIn(padding[:40], tail)
        self.assertIn("Fix the plugin", tail)
        self.assertEqual(full_multi.count(padding), 1)
        # Primary user delivery in a delta never drops a user message starting
        # with [PRIOR CONTEXT even when no summary markers are present.
        raw_user_msg = "[PRIOR CONTEXT] Please check why my note had this prefix."
        primary_delta = _format_delta_prompt([{"role": "user", "content": raw_user_msg}])
        self.assertIn(raw_user_msg, primary_delta)
        # Multi-line quoted user request fallback preserves all consecutive > lines.
        multiline_summary = (
            "[CONTEXT COMPACTION]\n## User Messages (verbatim, newest first)\n"
            "> Line one\n> Line two\n\n## Next Section\n"
        )
        self.assertEqual(_latest_user_text([{"role": "user", "content": multiline_summary}]), "Line one\nLine two")

    def test_prior_context_message_without_markers_keeps_its_text_in_tool_turns(self):
        # A user message that merely starts with the prefix is not a compaction
        # summary: tool-result and full prompts must not restate nothing.
        raw = "[PRIOR CONTEXT] Please check why my note had this prefix."
        history = [
            {"role": "user", "content": raw},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "supply.db 108K"},
        ]
        self.assertEqual(_latest_user_text(history), raw)

        delta = _format_delta_prompt(history[-2:], pending_user_text=_latest_user_text(history))
        self.assertIn(raw, delta)

        full = _format_messages_as_prompt(history)
        head = full[: full.index("Conversation transcript:")]
        self.assertIn("### ACTIVE OBJECTIVE:", head)
        self.assertIn(raw, head)

    def test_summary_extraction_spans_blank_lines_between_quoted_lines(self):
        summary = (
            "[CONTEXT COMPACTION]\n## User Messages (verbatim, newest first)\n"
            "> Fix the plugin\n\n> and rerun the tests\n\n## Next Section\n"
        )
        self.assertEqual(
            _latest_user_text([{"role": "user", "content": summary}]),
            "Fix the plugin\nand rerun the tests",
        )

    def test_unparseable_summary_falls_back_to_bounded_text(self):
        # A summary with markers but no extractable ask must not restate nothing.
        summary = "[CONTEXT COMPACTION] keep this ask\n--- END OF CONTEXT SUMMARY ---\n"
        self.assertIn("keep this ask", _latest_user_text([{"role": "user", "content": summary}]))


if __name__ == "__main__":
    unittest.main()
