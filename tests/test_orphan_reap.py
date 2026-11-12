"""Orphaned plugin-worker reaper: scoped, Linux-only, owner-tempdir keyed.

A Hermes process that dies without close() leaves its agy worker orphaned
(PPID=1), holding a conversation open on the shared Google account — its
turns then tangle with every new session's turns (observed 2026-11-11,
hermes_agy_* homes + a HOME=/root orphan from 06:36). Pins that the sweep:

- kills a stream-json agy worker whose HOME sits inside a DELETED
  hermes_agy_* tempdir (the owner client is certainly gone);
- never touches the same worker while its owner tempdir still exists;
- never touches a worker with HOME outside a hermes_agy_* tempdir
  (interactive agy, other sessions' real-home workers).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from process import reap_orphaned_plugin_workers  # noqa: E402

if not sys.platform.startswith("linux"):
    raise unittest.SkipTest("orphan sweep is Linux-only (/proc)")

_WORKER_ARGV = [
    "--input-format",
    "stream-json",
    "--disable-slash-commands",
]


class _FakeWorker:
    def __init__(self, home: str):
        env = dict(os.environ)
        env["HOME"] = home
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", *_WORKER_ARGV],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_gone(self, seconds: float = 5.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not self.alive():
                return True
            time.sleep(0.1)
        self.proc.kill()
        return False


class OrphanReapTests(unittest.TestCase):
    def _mk_marker_home(self) -> str:
        base = tempfile.mkdtemp(prefix="hermes_agy_test_")
        return base

    def setUp(self):
        self._bases: list[str] = []

    def tearDown(self):
        for base in self._bases:
            shutil.rmtree(base, ignore_errors=True)

    def _home_under_marker(self) -> str:
        base = self._mk_marker_home()
        self._bases.append(base)
        home = os.path.join(base, "home")
        os.makedirs(home)  # like setup_isolated_home: the owner dir exists
        return home

    def test_orphan_with_deleted_tempdir_reaped(self):
        base = self._mk_marker_home()
        self._bases.append(base)
        home = os.path.join(base, "home")
        worker = _FakeWorker(home)
        self.assertTrue(worker.alive())
        shutil.rmtree(base, ignore_errors=True)  # owner gone
        reaped = reap_orphaned_plugin_workers()
        self.assertGreaterEqual(reaped, 1)
        self.assertTrue(worker.wait_gone())

    def test_worker_with_live_tempdir_untouched(self):
        home = self._home_under_marker()
        worker = _FakeWorker(home)
        try:
            reap_orphaned_plugin_workers()
            self.assertTrue(worker.alive())
        finally:
            worker.proc.kill()
            worker.proc.wait(timeout=5)

    def test_real_home_worker_untouched(self):
        home = tempfile.mkdtemp(prefix="plain_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        worker = _FakeWorker(home)  # no hermes_agy_ marker in HOME
        try:
            reap_orphaned_plugin_workers()
            self.assertTrue(worker.alive())
        finally:
            worker.proc.kill()
            worker.proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
