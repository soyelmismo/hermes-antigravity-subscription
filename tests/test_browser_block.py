"""Probes must never be able to open a browser window.

`agy` opens the system browser when it decides a session needs re-login. Every
read-only probe (list, /usage, status bar, desktop, rotation picks) runs
unattended, so a probe that re-authenticates would pop a browser tab the user
never asked for. These tests pin the guard that prevents it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import accounts  # noqa: E402


class BrowserBlockTests(unittest.TestCase):
    def test_probe_env_blocks_browser_by_default(self):
        env = accounts.probe_env("/tmp")
        self.assertEqual(env["BROWSER"], "/usr/bin/false")
        shim_dir = env["PATH"].split(os.pathsep)[0]
        shim = os.path.join(shim_dir, "open")
        self.assertTrue(os.path.isfile(shim))
        self.assertTrue(os.access(shim, os.X_OK))
        self.assertEqual(env["HOME"], str(Path("/tmp").resolve()))

    def test_block_can_be_disabled_for_interactive_sign_in(self):
        with patch.dict(os.environ, {"ANTIGRAVITY_BLOCK_BROWSER": "0"}):
            env = accounts.probe_env("/tmp")
        self.assertNotEqual(env.get("BROWSER"), "/usr/bin/false")
        self.assertNotIn("agy-nobrowser", env.get("PATH", ""))

    def test_shim_refuses_to_launch_a_browser(self):
        env = accounts.probe_env("/tmp")
        shim = os.path.join(env["PATH"].split(os.pathsep)[0], "open")
        res = subprocess.run([shim, "https://accounts.google.com/"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("blocked browser launch", res.stderr)

if __name__ == "__main__":
    unittest.main()
