"""Full-send segmentation for the Gemini single-message ceiling.

agy silently cuts one long Gemini user message and keeps only its head,
while one conversation accepts many such messages in a row. A full send that
does not fit is therefore split into consecutive turns of one agy
conversation: every part but the last asks for an ACK that is consumed, never
surfaced. These tests pin the splitter (boundaries, oversized messages, the
request in the last part) and the stream/client protocol (ACK discard, usage
of the final call only, Stop between parts, errors and native tools while
priming, the oneshot path).
"""

from collections import deque
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client as client_module  # noqa: E402
import prompt as prompt_module  # noqa: E402
import custom_agent  # noqa: E402
import stream as stream_module  # noqa: E402
from client import AntigravityClient  # noqa: E402
from prompt import (  # noqa: E402
    _format_messages_as_prompt,
    _format_messages_as_segments,
    _prompt_weight,
    _split_oversized_block,
    full_send_segment_budget,
)

GEMINI = "gemini-4-neon-low"
SMALL_BUDGET = 4_000


def _filler(tag: str, lines: int) -> str:
    return "\n".join(f"{tag} line {n:04d}: inventory note ref={n * 7919 % 100000}" for n in range(lines))


def _history() -> list[dict]:
    return [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "HEAD-CODE alpha-17\n" + _filler("u1", 40)},
        {"role": "assistant", "content": "Reading the file.", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "/tmp/x", "note": "n" * 600})},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "MID-CODE beta-42\n" + _filler("t1", 40)},
        {"role": "assistant", "content": "Done reading. " + _filler("a1", 30)},
        {"role": "user", "content": "TAIL-CODE gamma-99. Which three codes did you see?"},
    ]


class SplitterTests(unittest.TestCase):
    def test_prompt_that_fits_is_one_identical_message(self):
        messages = _history()
        full = _format_messages_as_prompt(messages, model=GEMINI)
        self.assertEqual(_format_messages_as_segments(messages, model=GEMINI, budget=None), [full])
        self.assertEqual(
            _format_messages_as_segments(messages, model=GEMINI, budget=_prompt_weight(full)), [full]
        )

    def test_overflow_splits_at_message_boundaries_and_keeps_order(self):
        messages = _history()
        segments = _format_messages_as_segments(messages, model=GEMINI, budget=SMALL_BUDGET)
        self.assertGreater(len(segments), 2)
        for segment in segments:
            self.assertLessEqual(_prompt_weight(segment), SMALL_BUDGET)
        total = len(segments)
        for index, segment in enumerate(segments[:-1], start=1):
            self.assertTrue(segment.startswith(f"[CONTEXT PART {index}/{total}]"))
            self.assertIn("Reply with exactly: ACK", segment.rsplit("\n", 1)[-1])
        self.assertTrue(segments[-1].startswith(f"[CONTEXT PART {total}/{total}, FINAL]"))
        joined = "\n".join(segments)
        # Every message survives, in order.
        positions = [joined.index(code) for code in ("HEAD-CODE alpha-17", "MID-CODE beta-42", "TAIL-CODE gamma-99")]
        self.assertEqual(positions, sorted(positions))
        for n in range(40):
            self.assertIn(f"u1 line {n:04d}", joined)
            self.assertIn(f"t1 line {n:04d}", joined)

    def test_tool_call_frame_is_never_cut(self):
        segments = _format_messages_as_segments(_history(), model=GEMINI, budget=SMALL_BUDGET)
        opener = '<tool_call>{"id": "call_1"'
        holders = [s for s in segments if opener in s]
        self.assertEqual(len(holders), 1)
        frame = holders[0][holders[0].index(opener):]
        self.assertIn("</tool_call>", frame)
        json.loads(frame[len("<tool_call>"):frame.index("</tool_call>")])

    def test_latest_request_and_closing_instruction_are_in_the_last_part(self):
        segments = _format_messages_as_segments(_history(), model=GEMINI, budget=SMALL_BUDGET)
        self.assertIn("### LATEST USER REQUEST TO ANSWER:", segments[-1])
        self.assertIn("### ACTIVE OBJECTIVE:", segments[-1])
        self.assertIn("TAIL-CODE gamma-99", segments[-1])
        for segment in segments[:-1]:
            self.assertNotIn("### LATEST USER REQUEST TO ANSWER:", segment)
            self.assertNotIn("### ACTIVE OBJECTIVE:", segment)
        # The preamble and system instructions lead the first part.
        self.assertIn("HEADLESS INFERENCE MODE", segments[0])
        self.assertIn("You are a careful assistant.", segments[0])

    def test_single_oversized_message_is_split_by_lines_with_markers(self):
        big = _filler("big", 400)
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": big},
            {"role": "user", "content": "what was the last line?"},
        ]
        segments = _format_messages_as_segments(messages, model=GEMINI, budget=SMALL_BUDGET)
        for segment in segments:
            self.assertLessEqual(_prompt_weight(segment), SMALL_BUDGET)
        joined = "\n".join(segments)
        self.assertIn("[... this message continues in piece 2/", joined)
        self.assertIn("[... continued: piece 2/", joined)
        for n in range(400):
            self.assertIn(f"big line {n:04d}:", joined)
        # Lines are never cut: each filler line appears whole exactly once.
        for n in (0, 199, 399):
            line = f"big line {n:04d}: inventory note ref={n * 7919 % 100000}"
            self.assertEqual(joined.count(line), 1)

    def test_single_line_heavier_than_budget_is_split_by_characters(self):
        pieces = _split_oversized_block("x" * 10_000, 3_000)
        self.assertEqual("".join(p.split("\n")[1] if p.startswith("[...") else p.split("\n")[0] for p in pieces), "x" * 10_000)
        for piece in pieces:
            self.assertLessEqual(_prompt_weight(piece), 3_000 + 120)

    def test_non_ascii_counts_double(self):
        self.assertEqual(_prompt_weight("abc"), 3)
        self.assertEqual(_prompt_weight("暗号abc"), 7)
        cjk = [
            {"role": "user", "content": "开始"},
            {"role": "assistant", "content": "\n".join("第%d行：库存记录，编号%d。" % (n, n * 31) for n in range(300))},
            {"role": "user", "content": "最后一行是什么？"},
        ]
        for segment in _format_messages_as_segments(cjk, model=GEMINI, budget=SMALL_BUDGET):
            self.assertLessEqual(_prompt_weight(segment), SMALL_BUDGET)

    def test_budget_is_per_family_and_env_overridable(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in prompt_module._SEGMENT_WEIGHT_ENV_VARS:
                os.environ.pop(var, None)
            self.assertEqual(full_send_segment_budget("gemini-4-neon-low"), prompt_module._GEMINI_SEGMENT_WEIGHT)
            self.assertEqual(full_send_segment_budget("gemini-3.8-flash-high"), prompt_module._GEMINI_SEGMENT_WEIGHT)
            self.assertIsNone(full_send_segment_budget("claude-opus-5-5-high"))
            os.environ["HERMES_ANTIGRAVITY_SEGMENT_WEIGHT"] = "50000"
            self.assertEqual(full_send_segment_budget("claude-opus-5-5-high"), 50_000)
            os.environ["HERMES_ANTIGRAVITY_SEGMENT_WEIGHT"] = "10"
            self.assertEqual(full_send_segment_budget(GEMINI), prompt_module._MIN_SEGMENT_WEIGHT)
            os.environ["HERMES_ANTIGRAVITY_SEGMENT_WEIGHT"] = "0"
            self.assertIsNone(full_send_segment_budget(GEMINI))
            os.environ.pop("HERMES_ANTIGRAVITY_SEGMENT_WEIGHT")


# --- protocol through the client ---------------------------------------------


class _Input:
    def __init__(self, owner):
        self.owner = owner
        self.writes: list[str] = []
        self.closed = False

    def write(self, value):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if not self.owner.alive:
            raise BrokenPipeError("dead")
        self.writes.append(value)
        if self.owner.on_write:
            self.owner.on_write(len(self.writes))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _TurnProcess:
    """Serves turn N's stdout only after the N-th stdin write, like agy does."""

    pid = None

    def __init__(self, turns):
        self.turns = deque(turns)
        self.stdin = _Input(self)
        self.stdout = self
        self.stderr = io.StringIO()
        self.buffer: deque[str] = deque()
        self.served = 0
        self.alive = True
        self.on_write = None

    def readline(self):
        if not self.alive:
            return ""
        if not self.buffer and self.served < len(self.stdin.writes) and self.turns:
            self.buffer.extend(json.dumps(e) + "\n" for e in self.turns.popleft())
            self.served += 1
        return self.buffer.popleft() if self.buffer else ""

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        return 0


def _turn(text, usage, conversation_id="conv-seg", *, status="SUCCESS", error=None, tool=None):
    events = [{"event": "init", "conversation_id": conversation_id}]
    if tool:
        events.append({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": tool, "tool_info": "x"}})
    events.append({"event": "step_update", "step_update": {"text_delta": text}})
    result = {"status": status, "response": text, "usage": usage}
    if error:
        result["error"] = error
    events.append({"event": "result", "result": result})
    return events


def _cumulative(n_parts, final_answer="codes: alpha-17 beta-42 gamma-99"):
    """Turn scripts with cumulative usage: part i reads 1000*i new tokens."""
    turns, cum_in, cum_out = [], 0, 0
    for i in range(1, n_parts + 1):
        cum_in += 1000 * i
        cum_out += 3
        text = "ACK" if i < n_parts else final_answer
        turns.append(_turn(text, {"input_tokens": cum_in, "output_tokens": cum_out,
                                  "total_tokens": cum_in + cum_out, "cache_read_tokens": 0}))
    return turns


class SegmentedSendTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="agy-seg-fixture-")
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        self.pending: deque = deque()
        self.spawn_calls: list = []
        self.terminated: list = []
        seams = {
            "setup_isolated_home": lambda cwd: (self.workspace / "home", None),
            "reap_orphaned_plugin_workers": lambda: None,
            "is_authenticated": lambda: True,
            "resolve_agy_command": lambda: "synthetic-agy",
            "resolve_model_and_effort": lambda model, effort: (model or GEMINI, effort),
            "sync_isolated_token": lambda home: None,
            "start_stderr_drainer": lambda proc: None,
            "_own_process_group": lambda: {},
            "build_child_env": lambda home: {},
            "read_media_rules": lambda home: frozenset(),
            "_worker_cpu_seconds": lambda proc: None,
            "dump_prompt_debug": lambda **kwargs: None,
        }
        for name, replacement in seams.items():
            p = patch.object(client_module, name, replacement)
            p.start()
            self.addCleanup(p.stop)
        for name, replacement in (
            ("_terminate_process", staticmethod(self._terminate)),
            ("_arm_idle_timer_locked", lambda self: None),
            ("_purge_isolated_conversations", lambda self: None),
        ):
            p = patch.object(AntigravityClient, name, replacement)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(client_module.subprocess, "Popen", self._spawn)
        p.start()
        self.addCleanup(p.stop)
        p = patch.dict(os.environ, {"HERMES_ANTIGRAVITY_SEGMENT_WEIGHT": str(SMALL_BUDGET)})
        p.start()
        self.addCleanup(p.stop)
        self.n_parts = len(_format_messages_as_segments(_history(), model=GEMINI, budget=SMALL_BUDGET))
        self.assertGreater(self.n_parts, 2)

    def _spawn(self, args, **kwargs):
        self.spawn_calls.append(args)
        if not self.pending:
            raise AssertionError("unexpected spawn")
        return self.pending.popleft()

    def _terminate(self, proc):
        proc.alive = False
        self.terminated.append(proc)

    def _client(self):
        c = AntigravityClient(cwd=str(self.workspace))
        self.addCleanup(c.close)
        return c

    @staticmethod
    def _contents(writes):
        return [json.loads(w)["message"]["content"] for w in writes]

    def test_worker_sends_parts_in_order_and_discards_acks(self):
        proc = _TurnProcess(_cumulative(self.n_parts))
        self.pending.append(proc)
        c = self._client()
        chunks = list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
        self.assertEqual(text, "codes: alpha-17 beta-42 gamma-99")
        self.assertNotIn("ACK", text)
        self.assertEqual(
            self._contents(proc.stdin.writes),
            _format_messages_as_segments(_history(), model=GEMINI, budget=SMALL_BUDGET),
        )
        usage = [ch.usage for ch in chunks if getattr(ch, "usage", None) is not None][-1]
        # Only the final call: its input already spans every part.
        self.assertEqual(usage.prompt_tokens, 1000 * self.n_parts)
        self.assertEqual(usage.completion_tokens, 3)
        self.assertEqual(c._worker_conversation_id, "conv-seg")
        self.assertEqual(c._worker_history, _history())
        self.assertFalse(c._worker_lock.locked())
        self.assertNotIn(proc, self.terminated)

    def test_continuation_after_segmented_send_is_a_plain_delta(self):
        proc = _TurnProcess(_cumulative(self.n_parts) + [_turn("next", {"input_tokens": 99_999, "output_tokens": 99})])
        self.pending.append(proc)
        c = self._client()
        list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        follow = _history() + [{"role": "assistant", "content": "codes"}, {"role": "user", "content": "thanks"}]
        chunks = list(c.chat.completions.create(model=GEMINI, messages=follow, stream=True))
        self.assertEqual(len(proc.stdin.writes), self.n_parts + 1)
        self.assertNotIn("[CONTEXT PART", self._contents(proc.stdin.writes)[-1])
        usage = [ch.usage for ch in chunks if getattr(ch, "usage", None) is not None][-1]
        cum_in = sum(1000 * i for i in range(1, self.n_parts + 1))
        self.assertEqual(usage.prompt_tokens, 99_999 - cum_in)

    def test_nonstreaming_segmented_send(self):
        self.pending.append(_TurnProcess(_cumulative(self.n_parts)))
        c = self._client()
        completion = c.chat.completions.create(model=GEMINI, messages=_history(), stream=False)
        self.assertEqual(completion.choices[0].message.content, "codes: alpha-17 beta-42 gamma-99")
        self.assertEqual(completion.usage.prompt_tokens, 1000 * self.n_parts)

    def test_stop_between_parts_cancels_without_sending_more(self):
        proc = _TurnProcess(_cumulative(self.n_parts))
        self.pending.append(proc)
        c = self._client()

        def stop_on_second_part(count):
            if count == 2:
                c.cancel()

        proc.on_write = stop_on_second_part
        stream = c.chat.completions.create(model=GEMINI, messages=_history(), stream=True)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            for chunk in stream:
                self.assertFalse(chunk.choices and chunk.choices[0].delta.content)
        self.assertLessEqual(len(proc.stdin.writes), 2)
        self.assertFalse(proc.alive)
        self.assertIsNone(c._worker_proc)

    def test_priming_error_fails_the_request_with_the_agy_error(self):
        turns = _cumulative(self.n_parts)
        turns[1] = _turn("", {"input_tokens": 1}, status="ERROR", error="max_trajectory_tokens exceeded")
        proc = _TurnProcess(turns)
        self.pending.append(proc)
        c = self._client()
        with self.assertRaisesRegex(RuntimeError, "max_trajectory_tokens"):
            list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        self.assertEqual(len(proc.stdin.writes), 2)
        self.assertFalse(proc.alive)
        self.assertEqual(c._worker_history, [])

    def test_priming_usage_reported_only_on_a_step_still_advances_the_baseline(self):
        # agy may put a turn's usage on its step_update instead of the result.
        turns = _cumulative(self.n_parts)
        for i in range(self.n_parts - 1):
            result = turns[i][-1]["result"]
            usage = result.pop("usage")
            turns[i].insert(2, {"event": "step_update", "step_update": {"usage": usage}})
        proc = _TurnProcess(turns)
        self.pending.append(proc)
        c = self._client()
        chunks = list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        usage = [ch.usage for ch in chunks if getattr(ch, "usage", None) is not None][-1]
        self.assertEqual(usage.prompt_tokens, 1000 * self.n_parts)

    def test_an_agent_refused_during_priming_is_recorded(self):
        # A split full send fails in _finish_priming_turn, before the result
        # handler that records a refused agent. It must be recorded here too.
        turns = _cumulative(self.n_parts)
        turns[0] = _turn("", {"input_tokens": 1}, status="ERROR", error='unknown agent "hermes-inference-backend"')
        proc = _TurnProcess(turns)
        self.pending.append(proc)
        c = self._client()
        # The client names the agent itself; a spawn overwrites the flag.
        with patch.object(custom_agent, "agy_version", return_value=custom_agent.MIN_AGY_VERSION), \
                patch.object(stream_module, "note_custom_agent_rejection", return_value=True) as note:
            with self.assertRaisesRegex(RuntimeError, "unknown agent"):
                list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        self.assertEqual(note.call_count, 1)

    def test_native_tool_while_priming_aborts_and_kills_the_worker(self):
        turns = _cumulative(self.n_parts)
        turns[0] = _turn("ACK", {"input_tokens": 1000}, tool="run_command")
        proc = _TurnProcess(turns)
        self.pending.append(proc)
        c = self._client()
        stream = c.chat.completions.create(model=GEMINI, messages=_history(), stream=True)
        with self.assertRaisesRegex(RuntimeError, "native tool invocation 'run_command' while priming context part 1/"):
            for chunk in stream:
                self.assertFalse(chunk.choices and chunk.choices[0].delta.content)
        self.assertEqual(len(proc.stdin.writes), 1)
        self.assertFalse(proc.alive)

    def test_worker_dying_between_parts_reports_the_part(self):
        turns = _cumulative(self.n_parts)[:2]
        proc = _TurnProcess(turns)
        self.pending.append(proc)
        c = self._client()

        def die_on_third(count):
            if count == 3:
                proc.alive = False

        proc.on_write = die_on_third
        with self.assertRaisesRegex(RuntimeError, r"while priming context part 3/"):
            list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))

    def test_oneshot_segmented_send_uses_ndjson_and_closes_stdin(self):
        proc = _TurnProcess(_cumulative(self.n_parts))
        self.pending.append(proc)
        c = self._client()
        self.assertTrue(c._worker_lock.acquire(blocking=False))  # force the oneshot path
        try:
            chunks = list(c.chat.completions.create(model=GEMINI, messages=_history(), stream=True))
        finally:
            c._worker_lock.release()
        args = self.spawn_calls[-1]
        self.assertEqual(args[1:3], ["--input-format", "stream-json"])
        self.assertTrue(proc.stdin.closed)
        self.assertEqual(len(proc.stdin.writes), self.n_parts)
        text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
        self.assertEqual(text, "codes: alpha-17 beta-42 gamma-99")
        usage = [ch.usage for ch in chunks if getattr(ch, "usage", None) is not None][-1]
        self.assertEqual(usage.prompt_tokens, 1000 * self.n_parts)

    def test_oneshot_that_fits_keeps_plain_text_stdin(self):
        proc = _TurnProcess([_turn("hi", {"input_tokens": 5, "output_tokens": 1})])
        self.pending.append(proc)
        c = self._client()
        messages = [{"role": "user", "content": "short"}]
        self.assertTrue(c._worker_lock.acquire(blocking=False))
        try:
            list(c.chat.completions.create(model=GEMINI, messages=messages, stream=True))
        finally:
            c._worker_lock.release()
        self.assertNotIn("--input-format", self.spawn_calls[-1])
        self.assertEqual(proc.stdin.writes, [_format_messages_as_prompt(messages, model=GEMINI) + "\n"])

    def test_claude_route_is_never_split_without_override(self):
        os.environ.pop("HERMES_ANTIGRAVITY_SEGMENT_WEIGHT")
        proc = _TurnProcess([_turn("fine", {"input_tokens": 5, "output_tokens": 1})])
        self.pending.append(proc)
        c = self._client()
        big = _history()
        big[1] = {"role": "user", "content": _filler("huge", 6000)}
        list(c.chat.completions.create(model="claude-opus-5-5-high", messages=big, stream=True))
        self.assertEqual(len(proc.stdin.writes), 1)
        self.assertNotIn("[CONTEXT PART", self._contents(proc.stdin.writes)[0])


if __name__ == "__main__":
    unittest.main()
