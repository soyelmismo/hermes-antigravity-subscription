"""Tests for UTF-8 byte-cap prompt fitting and truncation-notice learning (issue #80)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import agy_context  # noqa: E402
import prompt  # noqa: E402


class ByteBudgetConfigTests(unittest.TestCase):
    def setUp(self):
        agy_context.reset_learned_cap()
        self.addCleanup(agy_context.reset_learned_cap)

    def test_default_budget_derives_from_measured_cap(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_ANTIGRAVITY_PROMPT_BYTES", None)
            os.environ.pop("ANTIGRAVITY_PROMPT_BYTES", None)
            self.assertEqual(
                agy_context.single_turn_byte_cap(),
                agy_context.MEASURED_SINGLE_TURN_BYTES,
            )
            self.assertEqual(
                agy_context.prompt_byte_budget(),
                int(agy_context.MEASURED_SINGLE_TURN_BYTES * agy_context.BUDGET_FRACTION),
            )

    def test_zero_override_disables_byte_budget(self):
        with patch.dict(os.environ, {"HERMES_ANTIGRAVITY_PROMPT_BYTES": "0"}):
            self.assertIsNone(agy_context.single_turn_byte_cap())
            self.assertIsNone(agy_context.prompt_byte_budget())

    def test_hermes_env_wins_over_antigravity_env(self):
        with patch.dict(
            os.environ,
            {
                "HERMES_ANTIGRAVITY_PROMPT_BYTES": "100000",
                "ANTIGRAVITY_PROMPT_BYTES": "50000",
            },
        ):
            self.assertEqual(agy_context.single_turn_byte_cap(), 100_000)
            self.assertEqual(agy_context.prompt_byte_budget(), 94_000)


class TruncationLearningTests(unittest.TestCase):
    def setUp(self):
        agy_context.reset_learned_cap()
        self.addCleanup(agy_context.reset_learned_cap)

    def test_parses_backtick_and_angle_bracket_notices(self):
        n1 = agy_context.parse_truncation_notice(
            "Your message was truncated after `Record 00405: T` (`228,961 bytes` truncated)"
        )
        self.assertIsNotNone(n1)
        assert n1 is not None
        self.assertEqual(n1.dropped_bytes, 228_961)

        n2 = agy_context.parse_truncation_notice("... <truncated 296125 bytes>")
        self.assertIsNotNone(n2)
        assert n2 is not None
        self.assertEqual(n2.dropped_bytes, 296_125)

        self.assertIsNone(agy_context.parse_truncation_notice("All good, no cut here."))

    def test_note_truncation_only_lowers_learned_cap(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_ANTIGRAVITY_PROMPT_BYTES", None)
            os.environ.pop("ANTIGRAVITY_PROMPT_BYTES", None)
            first = agy_context.note_truncation("<truncated 100000 bytes>", 250_000)
            assert first is not None
            self.assertEqual(first.observed_cap, 150_000)
            self.assertEqual(agy_context.learned_single_turn_bytes(), 150_000)
            self.assertEqual(agy_context.single_turn_byte_cap(), 150_000)

            # A larger surviving figure must never raise a stricter learned cap.
            agy_context.note_truncation("<truncated 20000 bytes>", 190_000)
            self.assertEqual(agy_context.learned_single_turn_bytes(), 150_000)

            # A smaller surviving figure lowers it further.
            agy_context.note_truncation("<truncated 80000 bytes>", 200_000)
            self.assertEqual(agy_context.learned_single_turn_bytes(), 120_000)

    def test_note_truncation_rejects_untrusted_and_implausible_caps(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_ANTIGRAVITY_PROMPT_BYTES", None)
            os.environ.pop("ANTIGRAVITY_PROMPT_BYTES", None)
            # Untrusted model reply (learn=False) must never poison the global cap.
            agy_context.note_truncation("<truncated 100000 bytes>", 250_000, learn=False)
            self.assertIsNone(agy_context.learned_single_turn_bytes())
            # Implausibly tiny surviving cap (< MIN_LEARNED_SINGLE_TURN_BYTES) is ignored.
            agy_context.note_truncation("<truncated 9999 bytes>", 10_000, learn=True)
            self.assertIsNone(agy_context.learned_single_turn_bytes())
            agy_context.note_truncation("<truncated 180000 bytes>", 200_000, learn=True)
            self.assertIsNone(agy_context.learned_single_turn_bytes())


class ShrinkLadderTests(unittest.TestCase):
    def setUp(self):
        agy_context.reset_learned_cap()
        self.addCleanup(agy_context.reset_learned_cap)

    def test_small_prompt_is_untouched(self):
        msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello world"},
        ]
        out = prompt._format_messages_as_prompt(msgs)
        self.assertIn("Hello world", out)
        self.assertNotIn("EARLIER CONTEXT OMITTED", out)

    def test_oversized_history_shrinks_within_byte_budget_and_keeps_live_request(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(60):
            msgs.append({"role": "user", "content": f"turn {i}: " + ("x" * 4_000)})
            msgs.append({"role": "assistant", "content": f"reply {i}: " + ("y" * 4_000)})
        msgs.append({"role": "user", "content": "TAILMARK-9083: final question"})

        with patch.dict(os.environ, {"HERMES_ANTIGRAVITY_PROMPT_BYTES": "40000"}):
            budget = agy_context.prompt_byte_budget()
            assert budget is not None
            rendered = prompt._format_messages_as_prompt(msgs)
            self.assertLessEqual(agy_context.prompt_bytes(rendered), budget)
            self.assertIn("TAILMARK-9083: final question", rendered)
            self.assertIn("EARLIER CONTEXT OMITTED", rendered)

    def test_zero_budget_disables_shrinking(self):
        msgs = [
            {"role": "user", "content": "HEADMARK-4417 " + ("a" * 220_000)},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "TAILMARK-9083"},
        ]
        with patch.dict(os.environ, {"HERMES_ANTIGRAVITY_PROMPT_BYTES": "0"}):
            rendered = prompt._format_messages_as_prompt(msgs)
            self.assertGreater(agy_context.prompt_bytes(rendered), 220_000)
            self.assertIn("HEADMARK-4417", rendered)
            self.assertIn("TAILMARK-9083", rendered)

    def test_oversized_delta_prompt_fits_byte_budget(self):
        delta_msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "log " + ("z" * 25_000)},
            {"role": "tool", "tool_call_id": "c2", "content": "log " + ("w" * 25_000)},
        ]
        with patch.dict(os.environ, {"HERMES_ANTIGRAVITY_PROMPT_BYTES": "16000"}):
            budget = agy_context.prompt_byte_budget()
            assert budget is not None
            rendered = prompt._format_delta_prompt(delta_msgs, pending_user_text="summarize logs")
            self.assertLessEqual(agy_context.prompt_bytes(rendered), budget)
            self.assertIn("summarize logs", rendered)

    def test_hard_trim_respects_multibyte_utf8_budget(self):
        huge = "START " + ("😀" * 5_000) + " END"
        trimmed = agy_context.hard_trim(huge, 4_096)
        self.assertLessEqual(agy_context.prompt_bytes(trimmed), 4_096)
        self.assertTrue(trimmed.startswith("START "))
        self.assertIn("omitted to fit the agy frame", trimmed)

    def test_shrink_levels_do_not_inflate_medium_historical_tool_outputs(self):
        msgs = [{"role": "system", "content": "Be brief."}]
        for i in range(12):
            msgs.append({"role": "assistant", "content": f"step {i}", "tool_calls": [{"id": f"c{i}", "function": {"name": "t", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "a" * 150})
        msgs.append({"role": "user", "content": "done"})
        lvl0 = prompt._join_full_prompt_parts(prompt._assemble_full_prompt_parts(msgs, level=prompt._SHRINK_LEVELS[0]))
        lvl1 = prompt._join_full_prompt_parts(prompt._assemble_full_prompt_parts(msgs, level=prompt._SHRINK_LEVELS[1]))
        lvl2 = prompt._join_full_prompt_parts(prompt._assemble_full_prompt_parts(msgs, level=prompt._SHRINK_LEVELS[2]))
        self.assertLessEqual(len(lvl1), len(lvl0))
        self.assertLessEqual(len(lvl2), len(lvl1))

    def test_segmented_full_send_respects_byte_budget(self):
        msgs = [{"role": "system", "content": "Be helpful."}]
        for i in range(15):
            msgs.append({"role": "user", "content": f"msg {i}: " + ("😀" * 800)})
            msgs.append({"role": "assistant", "content": f"ack {i}"})
        msgs.append({"role": "user", "content": "FINAL-ASK-77"})
        with patch.dict(os.environ, {"HERMES_ANTIGRAVITY_PROMPT_BYTES": "30000"}):
            byte_cap = agy_context.prompt_byte_budget()
            assert byte_cap is not None
            segments = prompt._format_messages_as_segments(msgs, model="gemini-4-neon", budget=90_000)
            self.assertGreaterEqual(len(segments), 1)
            for seg in segments:
                self.assertLessEqual(agy_context.prompt_bytes(seg), byte_cap)
            self.assertIn("FINAL-ASK-77", segments[-1])


if __name__ == "__main__":
    unittest.main()
