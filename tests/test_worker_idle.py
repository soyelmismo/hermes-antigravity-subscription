"""Tests for the idle bound on the persistent agy worker.

Hermes retires an evicted agent's client without calling ``close()``, so the
persistent worker of a finished conversation used to stay alive for the
lifetime of the gateway. The client now terminates a worker that has not
finished a turn for ``ANTIGRAVITY_WORKER_IDLE_SECONDS`` (default 900).
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import (
    _DEFAULT_TIMEOUT_SECONDS,
    _DEFAULT_WORKER_IDLE_SECONDS,
    _worker_abandon_seconds,
    _worker_idle_seconds,
    AntigravityClient,
)

TURN_MESSAGES = [{"role": "user", "content": "hello agy"}]
IDLE_ENV = "ANTIGRAVITY_WORKER_IDLE_SECONDS"
ABANDON_ENV = "ANTIGRAVITY_WORKER_ABANDON_SECONDS"

# Upper bound for a reap that is expected to happen; a regression fails the
# test instead of hanging it.
REAP_DEADLINE_SECONDS = 5.0


class WorkerIdleTests(unittest.TestCase):
    def setUp(self) -> None:
        # CPU time is the second signal that separates a silent-but-busy turn
        # from an abandoned one; drive it per test through self.cpu.
        self.cpu: dict = {}
        patchers = (
            patch("client.is_authenticated", return_value=True),
            patch("client._worker_cpu_seconds", side_effect=lambda proc: self.cpu.get(proc)),
            patch("process.resolve_real_token_path", return_value=None),
            patch("client.resolve_agy_command", return_value="agy"),
            patch("process._link_macos_keychains"),
            patch("client.terminate_process"),
        )
        for patcher in patchers:
            mock = patcher.start()
            self.addCleanup(patcher.stop)
        self.terminate_process = mock
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        self.client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(self.client.close)

    def _install_worker(self) -> MagicMock:
        proc = MagicMock()
        proc.poll.return_value = None
        self.client._worker_proc = proc
        self.client._active_processes.add(proc)
        self.cpu.setdefault(proc, 0.0)
        return proc

    def _wait_for_reap(self) -> None:
        deadline = time.monotonic() + REAP_DEADLINE_SECONDS
        while self.client._worker_proc is not None and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_idle_worker_is_terminated_after_the_bound(self):
        proc = self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
            self.client._update_worker_history(TURN_MESSAGES)
        self._wait_for_reap()
        self.assertIsNone(self.client._worker_proc)
        self.terminate_process.assert_called_once_with(proc)
        # The next turn must take the full-prompt branch on a fresh worker.
        self.assertEqual(self.client._worker_history, [])
        self.assertNotIn(proc, self.client._active_processes)

    def test_worker_with_a_running_turn_is_not_reaped(self):
        proc = self._install_worker()
        self.assertTrue(self.client._worker_lock.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
                self.client._update_worker_history(TURN_MESSAGES)
            timer = self.client._worker_idle_timer
            timer.join(REAP_DEADLINE_SECONDS)
            self.assertFalse(timer.is_alive())
            self.assertIs(self.client._worker_proc, proc)
            self.terminate_process.assert_not_called()
        finally:
            self.client._worker_lock.release()

    def test_finished_turn_restarts_the_countdown(self):
        self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
            first = self.client._worker_idle_timer
            self.client._update_worker_history(TURN_MESSAGES * 2)
            second = self.client._worker_idle_timer
        self.assertIsNot(first, second)
        self.assertTrue(first.finished.is_set(), "the previous countdown was not cancelled")
        self.assertFalse(second.finished.is_set())

    def test_terminating_the_worker_cancels_the_countdown(self):
        self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
        timer = self.client._worker_idle_timer
        self.client._terminate_worker()
        self.assertIsNone(self.client._worker_idle_timer)
        self.assertTrue(timer.finished.is_set())

    def test_stale_countdown_does_not_terminate_a_replaced_worker(self):
        stale_proc = MagicMock()
        current_proc = self._install_worker()
        self.client._reap_idle_worker(stale_proc)
        self.assertIs(self.client._worker_proc, current_proc)
        self.terminate_process.assert_not_called()
        self.assertFalse(self.client._worker_lock.locked())

    def test_non_positive_bound_disables_the_countdown(self):
        self._install_worker()
        for value in ("0", "-1"):
            with patch.dict(os.environ, {IDLE_ENV: value}):
                self.client._update_worker_history(TURN_MESSAGES)
            self.assertIsNone(self.client._worker_idle_timer)

    def test_no_countdown_without_a_worker(self):
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
        self.assertIsNone(self.client._worker_idle_timer)

    def test_bound_defaults_when_unset_or_invalid(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(IDLE_ENV, None)
            self.assertEqual(_worker_idle_seconds(), _DEFAULT_WORKER_IDLE_SECONDS)
        with patch.dict(os.environ, {IDLE_ENV: "not-a-number"}):
            self.assertEqual(_worker_idle_seconds(), _DEFAULT_WORKER_IDLE_SECONDS)
        with patch.dict(os.environ, {IDLE_ENV: " 120 "}):
            self.assertEqual(_worker_idle_seconds(), 120.0)

    def test_a_worker_that_never_finishes_a_turn_still_expires(self):
        """Armed at spawn, not only when a turn completes.

        The countdown used to be armed solely by _update_worker_history, so a
        worker whose first turn never completed -- an abandoned stream that was
        never closed, a cancelled request, or a capacity retry that dropped the
        generator -- carried no timer and no close() caller, and stayed alive
        for the lifetime of the gateway.
        """
        proc = MagicMock()
        proc.poll.return_value = None
        with patch("client.subprocess.Popen", return_value=proc):
            with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
                spawned = self.client._get_or_spawn_worker("gemini-3.8-flash-high", None)
                self.assertIs(spawned, proc)
                self.assertIsNotNone(
                    self.client._worker_idle_timer,
                    "spawn did not arm the idle countdown",
                )
        self._wait_for_reap()
        self.assertIsNone(self.client._worker_proc)
        self.terminate_process.assert_called_once_with(proc)

    def test_a_running_turn_rearms_instead_of_losing_the_countdown(self):
        """A held worker lock re-arms the timer; it must not end the countdown.

        Before the fix the contention branch returned without re-arming, so a
        turn that never finished left the worker with no timer at all -- the
        same unbounded leak by another route.
        """
        proc = self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
            self.assertTrue(self.client._worker_lock.acquire(blocking=False))
            try:
                self.client._update_worker_history(TURN_MESSAGES)
                first = self.client._worker_idle_timer
                self.assertIsNotNone(first)
                assert first is not None  # narrow for the type checker
                first.join(REAP_DEADLINE_SECONDS)
                self.assertFalse(first.is_alive())
                self.assertIs(self.client._worker_proc, proc, "a worker with a live turn was terminated")
                self.terminate_process.assert_not_called()
                second = self.client._worker_idle_timer
                self.assertIsNotNone(second, "the countdown was not re-armed after contention")
                self.assertIsNot(second, first)
            finally:
                self.client._worker_lock.release()
            # The re-armed countdown must still end the worker once the lock is
            # free. Held inside the env patch because each re-arm reads the
            # bound afresh.
            self._wait_for_reap()
        self.assertIsNone(self.client._worker_proc)
        self.terminate_process.assert_called_once_with(proc)

    def test_abandoned_turn_is_terminated_after_the_abandon_grace(self):
        """A turn that died holding the worker lock must not pin its worker.

        An abandoned stream releases the lock only when the client/stream
        reference cycle is collected, which a lingering reference can postpone
        indefinitely. The countdown then re-armed forever: three dashboard
        workers were measured still alive past the bound because of it.
        """
        proc = self._install_worker()
        # Silent for ten seconds AND flat on CPU: provably abandoned.
        self.client._worker_last_activity[proc] = time.monotonic() - 10
        self.client._worker_cpu_at_activity[proc] = self.cpu[proc]
        self.assertTrue(self.client._worker_lock.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {IDLE_ENV: "0.05", ABANDON_ENV: "0.05"}):
                self.client._update_worker_history(TURN_MESSAGES)
                timer = self.client._worker_idle_timer
                self.assertIsNotNone(timer)
                assert timer is not None  # narrow for the type checker
                timer.join(REAP_DEADLINE_SECONDS)
                self.assertFalse(timer.is_alive())
        finally:
            self.client._worker_lock.release()
        self.assertIsNone(self.client._worker_proc, "the abandoned worker was not terminated")
        self.terminate_process.assert_called_once_with(proc)
        self.assertNotIn(proc, self.client._worker_last_activity)

    def test_a_turn_that_keeps_producing_output_is_never_abandoned(self):
        proc = self._install_worker()
        self.client._worker_last_activity[proc] = time.monotonic()
        self.assertTrue(self.client._worker_lock.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {IDLE_ENV: "0.05", ABANDON_ENV: "60"}):
                self.client._update_worker_history(TURN_MESSAGES)
                timer = self.client._worker_idle_timer
                self.assertIsNotNone(timer)
                assert timer is not None  # narrow for the type checker
                timer.join(REAP_DEADLINE_SECONDS)
                self.assertIs(self.client._worker_proc, proc, "a live turn's worker was terminated")
                self.terminate_process.assert_not_called()
                self.assertIsNotNone(
                    self.client._worker_idle_timer,
                    "the countdown was not re-armed after a live turn held the lock",
                )
        finally:
            self.client._worker_lock.release()

    def test_output_after_a_stale_baseline_defers_the_abandon_check(self):
        proc = self._install_worker()
        self.client._worker_last_activity[proc] = time.monotonic() - 600
        self.client._note_worker_activity(proc)
        self.assertTrue(self.client._worker_lock.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {IDLE_ENV: "0.05", ABANDON_ENV: "5"}):
                self.client._update_worker_history(TURN_MESSAGES)
                timer = self.client._worker_idle_timer
                self.assertIsNotNone(timer)
                assert timer is not None  # narrow for the type checker
                timer.join(REAP_DEADLINE_SECONDS)
                self.assertIs(self.client._worker_proc, proc)
                self.terminate_process.assert_not_called()
        finally:
            self.client._worker_lock.release()

    def test_activity_stamp_ignores_a_worker_the_client_no_longer_owns(self):
        self._install_worker()
        stale = MagicMock()
        self.client._note_worker_activity(stale)
        self.assertNotIn(stale, self.client._worker_last_activity)

    def test_zero_abandon_grace_disables_the_second_opinion(self):
        proc = self._install_worker()
        self.client._worker_last_activity[proc] = time.monotonic() - 10_000
        with patch.dict(os.environ, {ABANDON_ENV: "0"}):
            self.assertFalse(self.client._worker_is_abandoned_locked(proc))

    def test_a_silent_worker_that_is_still_burning_cpu_is_spared(self):
        """A live turn may emit nothing for minutes while it thinks.

        Silence alone must never be enough: a long reasoning block or a generous
        caller-supplied read deadline produces exactly the same silence as an
        abandoned stream, and killing it loses the turn.
        """
        proc = self._install_worker()
        self.client._worker_last_activity[proc] = time.monotonic() - 10
        self.client._worker_cpu_at_activity[proc] = 1.0
        self.cpu[proc] = 9.0
        with patch.dict(os.environ, {ABANDON_ENV: "0.05"}):
            self.assertFalse(self.client._worker_is_abandoned_locked(proc))

    def test_abandonment_needs_a_measurable_cpu_baseline(self):
        """Unmeasurable CPU means unproven abandonment: keep the worker."""
        proc = self._install_worker()
        self.client._worker_last_activity[proc] = time.monotonic() - 10_000
        self.cpu[proc] = None
        with patch.dict(os.environ, {ABANDON_ENV: "0.05"}):
            self.assertFalse(self.client._worker_is_abandoned_locked(proc))

    def test_a_reused_worker_starts_its_turn_from_a_fresh_baseline(self):
        """Continuation turns must not inherit the previous turn's silence.

        A worker idle for nearly a full window and then handed a new turn would
        otherwise be reaped mid-turn, because the grace would be measured from
        the old turn's last output.
        """
        proc = self._install_worker()
        self.client._worker_model = "gemini-3.8-flash"
        self.client._worker_effort = None
        self.client._worker_last_activity[proc] = time.monotonic() - 1000
        self.cpu[proc] = 3.0
        with patch.dict(os.environ, {ABANDON_ENV: "0.05"}):
            reused = self.client._get_or_spawn_worker("gemini-3.8-flash", None)
            self.assertIs(reused, proc)
            self.assertLess(time.monotonic() - self.client._worker_last_activity[proc], 1.0)
            self.assertEqual(self.client._worker_cpu_at_activity[proc], 3.0)
            self.assertFalse(self.client._worker_is_abandoned_locked(proc))

    def test_abandon_grace_never_undercuts_the_read_deadline(self):
        with patch.dict(os.environ, {IDLE_ENV: "1"}):
            os.environ.pop(ABANDON_ENV, None)
            self.assertGreaterEqual(_worker_abandon_seconds(), _DEFAULT_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
