"""Regression tests for issue #50: orphaned agy workers and zombies.

Covers:
- ``_own_process_group`` arms PR_SET_PDEATHSIG on Linux so a dead parent
  takes its agy child with it.
- The interpreter-exit registry terminates still-running children even if
  close() paths were skipped, and never aborts on one stubborn child.
- ``_kill_process_tree`` / ``terminate_process`` reap killed children and
  clean the registry (no ``<defunct>`` entries left under a live worker).
"""

import os
import subprocess
import sys
import time
import unittest
from unittest import mock

import process
from process import (
    _kill_process_tree,
    _own_process_group,
    _terminate_tracked_processes,
    register_tracked_process,
    terminate_process,
    unregister_tracked_process,
)

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_proc(pid=4242, poll_value=None, wait_exc=None):
    proc = mock.Mock()
    proc.pid = pid
    proc.poll.return_value = poll_value
    if wait_exc is not None:
        proc.wait.side_effect = wait_exc
    else:
        proc.wait.return_value = 0
    return proc


class _RegistryIsolation:
    """Save/restore the module-level tracked registry around each test."""

    def setUp(self):
        self._tracked = process._TRACKED_PROCS
        self._snapshot = set(self._tracked)
        self._tracked.clear()

    def tearDown(self):
        self._tracked.clear()
        self._tracked.update(self._snapshot)


class TestOwnProcessGroup(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "PDEATHSIG is Linux-only")
    def test_linux_arms_pdeathsig_preexec(self):
        kwargs = _own_process_group()
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(callable(kwargs["preexec_fn"]))

    @unittest.skipIf(sys.platform == "linux", "only the non-Linux path")
    def test_non_linux_has_no_preexec(self):
        kwargs = _own_process_group()
        self.assertNotIn("preexec_fn", kwargs)

    @unittest.skipUnless(sys.platform == "linux", "real fork test")
    def test_pdeathsig_kills_orphaned_grandchild(self):
        """Real reproduction of issue #50: parent dies hard, grandchild must die."""
        child_code = (
            "import subprocess, sys, os\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from process import _own_process_group\n"
            "gc = subprocess.Popen(['sleep', '300'], stdout=subprocess.DEVNULL,"
            " **_own_process_group())\n"
            "print(gc.pid); sys.stdout.flush()\n"
            "os._exit(0)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code, _PLUGIN_ROOT],
            stdout=subprocess.PIPE,
            text=True,
        )
        gcpid = int(parent.communicate()[0].strip())
        survived = None
        for _ in range(50):
            try:
                os.kill(gcpid, 0)
                survived = True
            except ProcessLookupError:
                survived = False
                break
            time.sleep(0.1)
        self.assertIs(survived, False, f"grandchild {gcpid} survived parent death")


class TestTrackedProcessRegistry(_RegistryIsolation, unittest.TestCase):
    def test_register_unregister_roundtrip(self):
        proc = _fake_proc()
        register_tracked_process(proc)
        self.assertIn(proc, self._tracked)
        unregister_tracked_process(proc)
        self.assertNotIn(proc, self._tracked)

    def test_unregister_is_idempotent(self):
        proc = _fake_proc()
        unregister_tracked_process(proc)  # never registered: must not raise
        register_tracked_process(proc)
        unregister_tracked_process(proc)
        unregister_tracked_process(proc)

    def test_exit_hook_terminates_only_live_children(self):
        live = _fake_proc(poll_value=None)
        dead = _fake_proc(poll_value=0)
        register_tracked_process(live)
        register_tracked_process(dead)
        with mock.patch.object(process, "terminate_process") as term:
            _terminate_tracked_processes()
        term.assert_called_once_with(live)
        self.assertEqual(self._tracked, set())

    def test_exit_hook_survives_terminate_failure(self):
        bad = _fake_proc(poll_value=None)
        good = _fake_proc(poll_value=None)
        register_tracked_process(bad)
        register_tracked_process(good)
        calls = []

        def fake_terminate(proc):
            calls.append(proc)
            if proc is bad:
                raise RuntimeError("taskkill exploded")

        with mock.patch.object(process, "terminate_process", side_effect=fake_terminate):
            _terminate_tracked_processes()
        # Set iteration order is arbitrary: both live children must be
        # attempted even though the first one blew up.
        self.assertEqual(sorted(calls, key=id), sorted([bad, good], key=id))


class TestKillReapsZombie(_RegistryIsolation, unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_waits_and_unregisters(self):
        proc = _fake_proc(pid=999999)
        register_tracked_process(proc)
        with mock.patch("process.os.killpg") as killpg:
            _kill_process_tree(proc)
        killpg.assert_called_once()
        proc.wait.assert_called_once_with(timeout=2)
        self.assertNotIn(proc, self._tracked)

    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_wait_failure_still_unregisters(self):
        proc = _fake_proc(pid=999999, wait_exc=subprocess.TimeoutExpired(cmd="agy", timeout=2))
        register_tracked_process(proc)
        with mock.patch("process.os.killpg"):
            _kill_process_tree(proc)
        self.assertNotIn(proc, self._tracked)

    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_already_dead(self):
        proc = _fake_proc(poll_value=0)
        register_tracked_process(proc)
        _kill_process_tree(proc)
        proc.wait.assert_not_called()
        self.assertNotIn(proc, self._tracked)

    def test_terminate_process_unregisters_after_graceful_exit(self):
        proc = _fake_proc()
        register_tracked_process(proc)
        terminate_process(proc)
        self.assertNotIn(proc, self._tracked)

    def test_terminate_process_unregisters_after_fallback_kill(self):
        proc = _fake_proc(wait_exc=subprocess.TimeoutExpired(cmd="agy", timeout=2))
        register_tracked_process(proc)
        with mock.patch("process._kill_process_tree") as hard:
            terminate_process(proc)
        hard.assert_called_once_with(proc)
        self.assertNotIn(proc, self._tracked)


if __name__ == "__main__":
    unittest.main()
