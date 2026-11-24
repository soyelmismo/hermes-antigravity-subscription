"""Usage of the first turn of a worker resumed with --conversation.

A worker reaped while idle (or recycled for media permissions) keeps its agy
conversation; the next turn respawns it with ``--conversation``. agy's usage
counters on a resumed conversation continue from the session total, so with
an empty baseline that turn reported every earlier turn as its prompt.
Measured with agy 1.5.15 + gemini-4-neon-low: 71,563 input + 129,939 cached
for a 67,457-token context; 4,954,420 tokens in a long Hermes session, which
made Hermes compress a context that was nowhere near full.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import AntigravityClient  # noqa: E402
from test_worker_usage import (  # noqa: E402
    MESSAGES_TURN_1, MESSAGES_TURN_2, MESSAGES_TURN_3, MODEL,
    _mock_proc, _turn_events, _turn_lines, _usage_of,
)


_REAL_POPEN = subprocess.Popen


def _agy_popen(*procs):
    """Popen stand-in: agy spawns get the scripted fakes in order, anything
    else (the ps-based CPU probe) runs for real."""
    queue = list(procs)

    def popen(args, *a, **kw):
        if args and args[0] == "agy":
            return queue.pop(0)
        return _REAL_POPEN(args, *a, **kw)

    return popen


def _usage(inp, out, cached):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out, "cache_read_tokens": cached}


class ResumedWorkerUsageTests(unittest.TestCase):
    def setUp(self):
        for target, kw in (
            ("client.is_authenticated", {"return_value": True}),
            ("process.resolve_real_token_path", {"return_value": None}),
            ("client.resolve_agy_command", {"return_value": "agy"}),
            ("process._link_macos_keychains", {}),
        ):
            p = patch(target, **kw)
            p.start()
            self.addCleanup(p.stop)

    def _client(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(client.close)
        return client

    def _create(self, client, messages):
        return _usage_of(list(client.chat.completions.create(model=MODEL, messages=messages, stream=True)))

    def test_first_turn_after_idle_reap_reports_only_that_call(self):
        # The cumulative counters of the real run: turn 1 66,916 / 0 cached,
        # turn 2 +2,270 / +64,859, resumed turn 3 reports 71,563 / 129,939.
        client = self._client()
        proc1 = _mock_proc(_turn_lines(
            _turn_events("conv-1", "a1", _usage(66_916, 37, 0)),
            _turn_events("conv-1", "a2", _usage(69_186, 55, 64_859)),
        ))
        proc2 = _mock_proc(_turn_lines(
            _turn_events("conv-1", "a3", _usage(71_563, 130, 129_939)),
        ))
        with patch("subprocess.Popen", side_effect=_agy_popen(proc1, proc2)) as popen:
            u1 = self._create(client, MESSAGES_TURN_1)
            u2 = self._create(client, MESSAGES_TURN_2)
            client._reap_idle_worker(client._worker_proc)
            self.assertIsNone(client._worker_proc)
            u3 = self._create(client, MESSAGES_TURN_3)
            argv = [c.args[0] for c in popen.call_args_list if c.args[0][0] == "agy"][1]
            self.assertEqual(argv[argv.index("--conversation") + 1], "conv-1")
        self.assertEqual((u1.prompt_tokens, u2.prompt_tokens), (66_916, 2_270))
        self.assertEqual(u3.prompt_tokens, 2_377)
        self.assertEqual(u3.prompt_tokens_details.cached_tokens, 65_080)
        self.assertEqual(u3.completion_tokens, 75)
        self.assertIsNone(client._resume_usage_baseline)

    def test_resumed_conversation_counters_that_restart_fall_back_to_raw_values(self):
        # If agy ever restarted the counters on resume, values below the
        # carried snapshot read as a restart: reported as-is, never negative.
        client = self._client()
        proc1 = _mock_proc(_turn_lines(_turn_events("conv-1", "a1", _usage(5_000, 10, 0))))
        proc2 = _mock_proc(_turn_lines(_turn_events("conv-1", "a2", _usage(800, 9, 4_000))))
        with patch("subprocess.Popen", side_effect=_agy_popen(proc1, proc2)):
            self._create(client, MESSAGES_TURN_1)
            client._reap_idle_worker(client._worker_proc)
            u2 = self._create(client, MESSAGES_TURN_2)
        self.assertEqual(u2.prompt_tokens, 800)
        self.assertEqual(u2.prompt_tokens_details.cached_tokens, 4_000)

    def test_fresh_conversation_does_not_inherit_a_snapshot(self):
        client = self._client()
        proc1 = _mock_proc(_turn_lines(_turn_events("conv-1", "a1", _usage(5_000, 10, 0))))
        proc2 = _mock_proc(_turn_lines(_turn_events("conv-2", "b1", _usage(777, 7, 0))))
        with patch("subprocess.Popen", side_effect=_agy_popen(proc1, proc2)) as popen:
            self._create(client, MESSAGES_TURN_1)
            client._reap_idle_worker(client._worker_proc)
            # Not a continuation: the conversation is dropped and respawned fresh.
            u = self._create(client, [{"role": "user", "content": "unrelated"}])
            self.assertNotIn("--conversation", [c.args[0] for c in popen.call_args_list if c.args[0][0] == "agy"][1])
        self.assertEqual(u.prompt_tokens, 777)
        self.assertIsNone(client._resume_usage_baseline)

    def test_snapshot_is_a_copy_so_a_stale_stream_cannot_move_it(self):
        client = self._client()
        proc1 = _mock_proc(_turn_lines(_turn_events("conv-1", "a1", _usage(5_000, 10, 0))))
        with patch("subprocess.Popen", side_effect=_agy_popen(proc1)):
            self._create(client, MESSAGES_TURN_1)
        old = client._worker_usage_baseline
        client._reap_idle_worker(client._worker_proc)
        conv, saved = client._resume_usage_baseline
        self.assertEqual(conv, "conv-1")
        self.assertEqual(saved, old)
        self.assertIsNot(saved, old)
        old["input_tokens"] = 9_999_999
        self.assertEqual(saved["input_tokens"], 5_000)


if __name__ == "__main__":
    unittest.main()
