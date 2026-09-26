"""Regression tests for per-turn worker usage deltas (issue #3).

agy 1.2.10/11 persistent workers report *cumulative session* usage: the
observed sequence input 12233, 24673, 37261 with outputs 134, 209, 284 must
reach Hermes as per-turn deltas 12233, 12440, 12588 and 134, 75, 75, otherwise
Hermes context compression fires on every turn. These tests pin the worker
delta accounting, its baseline resets (respawn, termination, decreasing
counters, omitted optional fields), and oneshot isolation.
"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from stream import AntigravityStream, _counter_delta, _delta_worker_usage

MODEL = "gemini-3.8-flash-high"

# Cumulative usage exactly as reported by agy for three consecutive turns of
# one persistent worker session (the sequence from the bug report).
CUMULATIVE_TURN_USAGE = (
    {"input_tokens": 12233, "output_tokens": 134, "total_tokens": 12367, "cache_read_tokens": 0},
    {"input_tokens": 24673, "output_tokens": 209, "total_tokens": 24882, "cache_read_tokens": 0},
    {"input_tokens": 37261, "output_tokens": 284, "total_tokens": 37545, "cache_read_tokens": 0},
)
EXPECTED_TURN_DELTAS = (
    (12233, 134, 12367),
    (12440, 75, 12515),
    (12588, 75, 12663),
)

MESSAGES_TURN_1 = [{"role": "user", "content": "turn one"}]
MESSAGES_TURN_2 = MESSAGES_TURN_1 + [
    {"role": "assistant", "content": "answer one"},
    {"role": "user", "content": "turn two"},
]
MESSAGES_TURN_3 = MESSAGES_TURN_2 + [
    {"role": "assistant", "content": "answer two"},
    {"role": "user", "content": "turn three"},
]


def _turn_events(conversation_id: str, text: str, usage: dict) -> list[dict]:
    """stream-json events a single agy worker turn emits before its usage."""
    return [
        {"event": "init", "conversation_id": conversation_id},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {"event": "result", "result": {"status": "SUCCESS", "response": text, "usage": usage}},
    ]


def _turn_lines(*turns: list[dict]) -> list[str]:
    lines: list[str] = []
    for turn in turns:
        lines.extend(json.dumps(event) + "\n" for event in turn)
    lines.append("")
    return lines


def _mock_proc(lines: list[str], *, alive: bool = True) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None if alive else 0
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = lines
    return proc


def _usage_of(chunks) -> Any:
    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            return usage
    raise AssertionError("stream yielded no usage chunk")


class WorkerUsageDeltaTests(unittest.TestCase):
    """Unit tests for the pure cumulative-to-delta conversion."""

    def test_first_report_of_a_counter_is_its_full_value(self):
        baseline: dict[str, int] = {}
        self.assertEqual(_counter_delta({"input_tokens": 42}, baseline, "input_tokens"), 42)
        self.assertEqual(baseline, {"input_tokens": 42})

    def test_increasing_counter_reports_the_delta(self):
        baseline = {"input_tokens": 12233}
        self.assertEqual(_counter_delta({"input_tokens": 24673}, baseline, "input_tokens"), 12440)
        self.assertEqual(baseline, {"input_tokens": 24673})

    def test_decreasing_counter_reports_current_value_and_rebaselines(self):
        baseline = {"input_tokens": 37261}
        self.assertEqual(_counter_delta({"input_tokens": 90}, baseline, "input_tokens"), 90)
        self.assertEqual(baseline, {"input_tokens": 90})

    def test_missing_field_is_zero_and_leaves_baseline_untouched(self):
        baseline = {"input_tokens": 100, "output_tokens": 10}
        self.assertEqual(_counter_delta({}, baseline, "output_tokens"), 0)
        self.assertEqual(baseline, {"input_tokens": 100, "output_tokens": 10})

    def test_null_field_value_is_treated_as_zero(self):
        baseline: dict[str, int] = {}
        self.assertEqual(_counter_delta({"input_tokens": None}, baseline, "input_tokens"), 0)
        self.assertEqual(baseline, {"input_tokens": 0})

    def test_total_tokens_falls_back_to_input_plus_output_when_absent(self):
        baseline: dict[str, int] = {}
        deltas = _delta_worker_usage({"input_tokens": 100, "output_tokens": 10}, baseline)
        self.assertEqual(deltas.input_tokens, 100)
        self.assertEqual(deltas.output_tokens, 10)
        self.assertEqual(deltas.total_tokens, 110)
        self.assertNotIn("total_tokens", baseline)

    def test_total_tokens_is_deltaed_like_the_other_counters_when_present(self):
        baseline: dict[str, int] = {}
        _delta_worker_usage(
            {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}, baseline
        )
        deltas = _delta_worker_usage(
            {"input_tokens": 250, "output_tokens": 40, "total_tokens": 290}, baseline
        )
        self.assertEqual(deltas.total_tokens, 180)

    def test_omitted_fields_do_not_reset_unrelated_counters(self):
        baseline: dict[str, int] = {}
        _delta_worker_usage(
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 5}, baseline
        )
        deltas = _delta_worker_usage({"input_tokens": 250, "output_tokens": 40}, baseline)
        self.assertEqual(deltas.cache_read_tokens, 0)
        # The cache counter keeps its baseline through the omission.
        self.assertEqual(baseline["cache_read_tokens"], 5)


class WorkerTurnUsageTests(unittest.TestCase):
    """End-to-end tests through AntigravityClient.chat.completions.create."""

    def setUp(self):
        # Pin every host-dependent seam, same rationale as the main suite.
        patcher_auth = patch("client.is_authenticated", return_value=True)
        patcher_token = patch("process.resolve_real_token_path", return_value=None)
        patcher_cmd = patch("client.resolve_agy_command", return_value="agy")
        patcher_auth.start()
        patcher_token.start()
        patcher_cmd.start()
        self.addCleanup(patcher_auth.stop)
        self.addCleanup(patcher_token.stop)
        self.addCleanup(patcher_cmd.stop)

    def _client(self) -> AntigravityClient:
        # A real per-test temp dir as cwd on every platform: hardcoded
        # POSIX-only paths like /tmp resolve to `<drive>:\tmp` on Windows
        # (fragile, non-hermetic), and an explicit cwd is never cleaned by
        # close(), so the TemporaryDirectory does it instead. LIFO cleanup
        # order guarantees the client closes before its workspace disappears.
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(client.close)
        return client

    def test_consecutive_worker_turns_report_per_turn_deltas_streaming(self):
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                *(
                    _turn_events("conv-1", f"answer {i + 1}", usage)
                    for i, usage in enumerate(CUMULATIVE_TURN_USAGE)
                )
            )
        )
        with patch("subprocess.Popen", return_value=proc):
            for messages, expected in zip(
                (MESSAGES_TURN_1, MESSAGES_TURN_2, MESSAGES_TURN_3), EXPECTED_TURN_DELTAS
            ):
                stream = client.chat.completions.create(
                    model=MODEL, messages=messages, stream=True
                )
                usage = _usage_of(list(stream))
                self.assertEqual(usage.prompt_tokens, expected[0])
                self.assertEqual(usage.completion_tokens, expected[1])
                self.assertEqual(usage.total_tokens, expected[2])
        self.assertEqual(
            client._worker_usage_baseline,
            {"input_tokens": 37261, "output_tokens": 284, "total_tokens": 37545, "cache_read_tokens": 0},
        )

    def test_consecutive_worker_turns_report_per_turn_deltas_nonstreaming(self):
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                *(
                    _turn_events("conv-1", f"answer {i + 1}", usage)
                    for i, usage in enumerate(CUMULATIVE_TURN_USAGE)
                )
            )
        )
        with patch("subprocess.Popen", return_value=proc):
            for messages, expected in zip(
                (MESSAGES_TURN_1, MESSAGES_TURN_2, MESSAGES_TURN_3), EXPECTED_TURN_DELTAS
            ):
                res = client.chat.completions.create(
                    model=MODEL, messages=messages, stream=False
                )
                self.assertEqual(res.usage.prompt_tokens, expected[0])
                self.assertEqual(res.usage.completion_tokens, expected[1])
                self.assertEqual(res.usage.total_tokens, expected[2])
        self.assertEqual(
            client._worker_usage_baseline,
            {"input_tokens": 37261, "output_tokens": 284, "total_tokens": 37545, "cache_read_tokens": 0},
        )

    def test_worker_respawn_resets_usage_baseline(self):
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                _turn_events("conv-1", "answer one", {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220}),
                _turn_events("conv-2", "answer two", {"input_tokens": 55, "output_tokens": 7, "total_tokens": 62}),
            )
        )
        with patch("subprocess.Popen", return_value=proc):
            usage1 = _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL, messages=MESSAGES_TURN_1, stream=True
                    )
                )
            )
            self.assertEqual((usage1.prompt_tokens, usage1.completion_tokens), (200, 20))

            # A conversation that does not extend the worker history respawns
            # the worker: the new session starts from a clean baseline and its
            # first turn reports full usage, not a negative delta.
            res2 = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "unrelated conversation"}],
                stream=False,
            )
            self.assertEqual((res2.usage.prompt_tokens, res2.usage.completion_tokens), (55, 7))
            self.assertEqual(res2.usage.total_tokens, 62)
        self.assertEqual(
            client._worker_usage_baseline,
            {"input_tokens": 55, "output_tokens": 7, "total_tokens": 62},
        )

    def test_decreasing_counter_resets_only_that_counter(self):
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                _turn_events("conv-1", "answer one", {"input_tokens": 200, "output_tokens": 20}),
                # agy restarted its counters mid-session: input dropped while
                # output kept growing. Each counter is baselined independently.
                _turn_events("conv-1", "answer two", {"input_tokens": 90, "output_tokens": 45}),
            )
        )
        with patch("subprocess.Popen", return_value=proc):
            _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL, messages=MESSAGES_TURN_1, stream=True
                    )
                )
            )
            usage2 = _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL, messages=MESSAGES_TURN_2, stream=True
                    )
                )
            )
        self.assertEqual((usage2.prompt_tokens, usage2.completion_tokens), (90, 25))
        self.assertEqual(usage2.total_tokens, 115)  # fallback: 90 + 25
        self.assertEqual(client._worker_usage_baseline, {"input_tokens": 90, "output_tokens": 45})

    def test_omitted_optional_fields_do_not_reset_unrelated_counters(self):
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                _turn_events(
                    "conv-1", "answer one", {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 5}
                ),
                # cache_read_tokens omitted: input/output still delta, cache
                # reports 0 but keeps its baseline.
                _turn_events("conv-1", "answer two", {"input_tokens": 250, "output_tokens": 40}),
                # output_tokens omitted: input/output/cache keep their own
                # accounting instead of resetting.
                _turn_events("conv-1", "answer three", {"input_tokens": 400, "cache_read_tokens": 30}),
            )
        )
        with patch("subprocess.Popen", return_value=proc):
            usage1 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_1, stream=True)
                )
            )
            usage2 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_2, stream=True)
                )
            )
            usage3 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_3, stream=True)
                )
            )

        self.assertEqual(usage1.prompt_tokens, 100)
        self.assertEqual(usage1.completion_tokens, 10)
        self.assertEqual(usage1.prompt_tokens_details.cached_tokens, 5)
        self.assertEqual(usage1.total_tokens, 110)  # fallback: 100 + 10

        self.assertEqual(usage2.prompt_tokens, 150)
        self.assertEqual(usage2.completion_tokens, 30)  # 40 - 10
        self.assertEqual(usage2.prompt_tokens_details.cached_tokens, 0)
        self.assertEqual(usage2.total_tokens, 180)  # fallback: 150 + 30

        self.assertEqual(usage3.prompt_tokens, 150)
        self.assertEqual(usage3.completion_tokens, 0)
        self.assertEqual(usage3.prompt_tokens_details.cached_tokens, 25)  # 30 - 5
        self.assertEqual(usage3.total_tokens, 150)  # fallback: 150 + 0

        self.assertEqual(
            client._worker_usage_baseline,
            {"input_tokens": 400, "output_tokens": 40, "cache_read_tokens": 30},
        )

    def test_interrupted_worker_stream_resets_usage_baseline(self):
        client = self._client()
        session_a = _mock_proc(
            _turn_lines(
                _turn_events("conv-1", "answer one", {"input_tokens": 100, "output_tokens": 10}),
                # Turn two is abandoned mid-stream: its result/usage event is
                # never consumed, mirroring a real interrupted worker.
                _turn_events("conv-1", "answer two", {"input_tokens": 260, "output_tokens": 40}),
            )
        )
        session_b = _mock_proc(
            _turn_lines(
                _turn_events("conv-2", "answer three", {"input_tokens": 70, "output_tokens": 5})
            )
        )
        with patch("subprocess.Popen", side_effect=[session_a, session_b]):
            usage1 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_1, stream=True)
                )
            )
            self.assertEqual((usage1.prompt_tokens, usage1.completion_tokens), (100, 10))
            # The completed turn advanced the baseline to its cumulative
            # values. This assertion is what makes the reset check below
            # meaningful: against a raw-forwarding implementation the
            # baseline never populates, so "== {}" would pass even when no
            # reset happens at all.
            self.assertEqual(
                client._worker_usage_baseline,
                {"input_tokens": 100, "output_tokens": 10},
            )

            stream2 = client.chat.completions.create(
                model=MODEL, messages=MESSAGES_TURN_2, stream=True
            )
            next(stream2)
            stream2.close()
            # close() -> client._terminate_worker -> _terminate_worker_locked
            # replaces the baseline dict, so the respawned session starts
            # clean instead of inheriting the dead session's counters.
            self.assertEqual(client._worker_usage_baseline, {})

            # The respawned worker's first turn reports full usage.
            usage3 = _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": "fresh conversation"}],
                        stream=True,
                    )
                )
            )
            self.assertEqual((usage3.prompt_tokens, usage3.completion_tokens), (70, 5))

    def test_stream_abandoned_after_finish_chunk_advances_baseline(self):
        """Consumers that break at the finish_reason chunk must not lose usage.

        The common OpenAI pattern `for chunk in stream: if finish_reason:
        break` abandons the stream right at the finish-reason yield and never
        calls close(): garbage collection finalizes the suspended generator
        via GeneratorExit, so the trailing usage chunk is never received.
        The baseline must already hold turn one's cumulative snapshot when
        that yield happens (advanced inside the generator before emitting
        the chunk). If it waited for the usage chunk, turn one's tokens would
        leak into turn two's delta and re-introduce the over-counting these
        deltas exist to prevent.
        """
        client = self._client()
        proc = _mock_proc(
            _turn_lines(
                # Turn one is abandoned immediately after its finish chunk.
                _turn_events("conv-1", "answer one", CUMULATIVE_TURN_USAGE[0]),
                _turn_events("conv-1", "answer two", CUMULATIVE_TURN_USAGE[1]),
            )
        )
        turn_one_baseline = {
            "input_tokens": 12233,
            "output_tokens": 134,
            "total_tokens": 12367,
            "cache_read_tokens": 0,
        }
        with patch("subprocess.Popen", return_value=proc):
            stream = client.chat.completions.create(
                model=MODEL, messages=MESSAGES_TURN_1, stream=True
            )
            stream_generator = stream._generator
            consumed = []
            for chunk in stream:
                consumed.append(chunk)
                if any(
                    getattr(choice, "finish_reason", None)
                    for choice in getattr(chunk, "choices", [])
                ):
                    break
            # The consumer stopped at the finish chunk; the usage chunk that
            # follows it in the generator was never yielded or received.
            self.assertEqual(consumed[-1].choices[0].finish_reason, "stop")
            self.assertIsNone(consumed[-1].usage)

            # The baseline already advanced to turn one's cumulative usage
            # even though the usage chunk was never consumed.
            self.assertEqual(client._worker_usage_baseline, turn_one_baseline)

            # Drop the reference WITHOUT close(), as a consumer that moved on
            # would. Garbage collection later finalizes the suspended
            # generator by raising GeneratorExit at its finish-reason yield;
            # while the consuming frame is still on the generator's f_back
            # chain that cannot happen yet, so the test performs that same
            # deferred finalization explicitly (note: the *generator's*
            # close, not the stream's close() -- the latter is the
            # interrupted/terminate path pinned by the test above).
            del stream
            stream_generator.close()
            # The turn had succeeded, so finalization keeps the worker (and
            # its advanced baseline) alive instead of terminating it.
            self.assertIsNotNone(client._worker_proc)
            self.assertEqual(client._worker_usage_baseline, turn_one_baseline)

            # The next turn deltas against the advanced baseline, instead of
            # absorbing the abandoned turn's cumulative usage in full.
            usage2 = _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL, messages=MESSAGES_TURN_2, stream=True
                    )
                )
            )
            self.assertEqual(
                (usage2.prompt_tokens, usage2.completion_tokens, usage2.total_tokens),
                EXPECTED_TURN_DELTAS[1],  # (12440, 75, 12515): not (24673, 75, 24882)
            )
            self.assertEqual(
                client._worker_usage_baseline,
                {"input_tokens": 24673, "output_tokens": 209, "total_tokens": 24882, "cache_read_tokens": 0},
            )

    def test_concurrent_oneshot_fallback_does_not_touch_worker_baseline(self):
        client = self._client()
        worker_proc = _mock_proc(
            _turn_lines(
                _turn_events("conv-1", "answer one", {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}),
                _turn_events("conv-1", "answer two", {"input_tokens": 250, "output_tokens": 40, "total_tokens": 290}),
            )
        )
        oneshot_proc = _mock_proc(
            _turn_lines(
                # Cumulative-looking usage from an unrelated oneshot process.
                _turn_events(
                    "oneshot-conv", "oneshot answer", {"input_tokens": 5000, "output_tokens": 500, "total_tokens": 5500}
                )
            ),
            alive=False,
        )
        with patch("subprocess.Popen", side_effect=[worker_proc, oneshot_proc]):
            usage1 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_1, stream=True)
                )
            )
            self.assertEqual(
                (usage1.prompt_tokens, usage1.completion_tokens, usage1.total_tokens),
                (100, 10, 110),
            )
            baseline_after_worker_turn = dict(client._worker_usage_baseline)
            self.assertEqual(
                baseline_after_worker_turn,
                {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            )

            # A concurrent request falls back to oneshot while the worker lock
            # is held by the in-flight worker stream.
            self.assertTrue(client._worker_lock.acquire(blocking=False))
            oneshot_res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "parallel question"}],
                stream=False,
            )
            # Oneshot usage is per-turn already: forwarded raw, untouched.
            self.assertEqual(oneshot_res.usage.prompt_tokens, 5000)
            self.assertEqual(oneshot_res.usage.completion_tokens, 500)
            self.assertEqual(oneshot_res.usage.total_tokens, 5500)
            self.assertEqual(client._worker_usage_baseline, baseline_after_worker_turn)
            client._worker_lock.release()

            # The next worker turn still deltas against the pre-oneshot baseline.
            usage2 = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_2, stream=True)
                )
            )
            self.assertEqual(
                (usage2.prompt_tokens, usage2.completion_tokens, usage2.total_tokens),
                (150, 30, 180),
            )

    def test_stale_worker_stream_cannot_reset_new_session_baseline(self):
        client = self._client()
        session_a = _mock_proc(
            _turn_lines(
                _turn_events("conv-a", "answer a", {"input_tokens": 12233, "output_tokens": 134}),
                # A late turn that a stale stream from session A consumes
                # after session B has already taken over.
                _turn_events("conv-a", "late answer", {"input_tokens": 13000, "output_tokens": 200}),
            )
        )
        session_b = _mock_proc(
            _turn_lines(
                _turn_events("conv-b", "answer b", {"input_tokens": 800, "output_tokens": 90})
            )
        )
        with patch("subprocess.Popen", side_effect=[session_a, session_b]):
            usage_a = _usage_of(
                list(
                    client.chat.completions.create(model=MODEL, messages=MESSAGES_TURN_1, stream=True)
                )
            )
            self.assertEqual((usage_a.prompt_tokens, usage_a.completion_tokens), (12233, 134))
            stale_baseline = client._worker_usage_baseline
            self.assertEqual(stale_baseline, {"input_tokens": 12233, "output_tokens": 134})

            # Session A dies and is replaced: the baseline is a fresh dict.
            client._terminate_worker()
            self.assertEqual(client._worker_usage_baseline, {})

            usage_b = _usage_of(
                list(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": "brand new conversation"}],
                        stream=True,
                    )
                )
            )
            self.assertEqual((usage_b.prompt_tokens, usage_b.completion_tokens), (800, 90))
            session_b_baseline = client._worker_usage_baseline
            self.assertEqual(session_b_baseline, {"input_tokens": 800, "output_tokens": 90})

        # A stream still holding session A's baseline deltas against that
        # session and cannot corrupt session B's accounting.
        stale_stream = AntigravityStream(
            proc=session_a,
            client=client,
            model=MODEL,
            timeout=30.0,
            is_worker=True,
            messages=MESSAGES_TURN_1,
            usage_baseline=stale_baseline,
        )
        stale_usage = _usage_of(list(stale_stream))
        self.assertEqual((stale_usage.prompt_tokens, stale_usage.completion_tokens), (767, 66))
        # The baseline keeps the latest cumulative snapshot, not the delta.
        self.assertEqual(stale_baseline, {"input_tokens": 13000, "output_tokens": 200})
        self.assertIsNot(session_b_baseline, stale_baseline)
        self.assertEqual(session_b_baseline, {"input_tokens": 800, "output_tokens": 90})


if __name__ == "__main__":
    unittest.main()
