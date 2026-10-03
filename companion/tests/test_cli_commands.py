"""Tests for the companion CLI commands (list layout, use, mode, ignite) and the registry contract."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import accounts  # noqa: E402
from cli import _cmd_add, _cmd_ignite, _cmd_list, _cmd_mode, _cmd_use, antigravity_command  # noqa: E402


class RegistryContractTests(unittest.TestCase):
    """The provider plugin stores/reads active_account as a plain label string."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = Path(self.tmp.name) / "accounts.json"
        self.env = patch.dict(
            os.environ,
            {
                "ANTIGRAVITY_ACCOUNTS_FILE": str(self.registry),
                "ANTIGRAVITY_ACCOUNTS_DIR": str(Path(self.tmp.name) / "dir"),
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_set_active_label_writes_a_string(self):
        accounts.set_active_label("giovani")
        self.assertEqual(accounts.load_registry()["active_account"], "giovani")
        self.assertEqual(accounts.active_store_name(), "giovani")

    def test_set_active_label_none_or_host_clears(self):
        accounts.set_active_label("giovani")
        accounts.set_active_label(None)
        self.assertNotIn("active_account", accounts.load_registry())

        accounts.set_active_label("giovani")
        accounts.set_active_label("host")
        self.assertNotIn("active_account", accounts.load_registry())
        self.assertEqual(accounts.active_store_name(), "")

    def test_legacy_dict_shape_still_reads(self):
        self.registry.write_text('{"version": 1, "accounts": [], "active_account": {"label": "old"}}')
        self.assertEqual(accounts.active_store_name(), "old")

    def test_registry_writes_preserve_unknown_fields(self):
        self.registry.write_text('{"version": 1, "accounts": [], "note": "keep me"}')
        accounts.save_rotation_mode("quota")
        accounts.save_quota_ignition(True)
        accounts.set_active_label("acc")
        data = accounts.load_registry()
        self.assertEqual(data["note"], "keep me")
        self.assertEqual(data["rotation_mode"], "quota")
        self.assertIs(data["quota_ignition"], True)
        self.assertEqual(data["active_account"], "acc")


class HelpersTests(unittest.TestCase):
    def test_format_time_remaining(self):
        self.assertEqual(accounts.format_time_remaining(4 * 3600 + 12 * 60), "in 4h 12m")
        self.assertEqual(accounts.format_time_remaining(3 * 86400 + 2 * 3600), "in 3d 2h")
        self.assertEqual(accounts.format_time_remaining(45), "in 45s")
        self.assertEqual(accounts.format_time_remaining(0), "now")

    def test_parse_reset_time(self):
        self.assertIsNotNone(accounts.parse_reset_time("2026-10-10T11:27:09Z"))
        self.assertIsNotNone(accounts.parse_reset_time(1760000000000))  # milliseconds
        self.assertIsNone(accounts.parse_reset_time("not-a-date"))

    def test_describe_rotation_mode_and_ignite_effect(self):
        self.assertIn("healthiest", accounts.describe_rotation_mode("quota"))
        self.assertIn("active account", accounts.describe_rotation_mode("fixed"))
        self.assertIn("idle", accounts.ignite_effect(True))
        self.assertIn("untouched", accounts.ignite_effect(False))


class ListCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = Path(self.tmp.name) / "accounts.json"
        self.accounts = [
            {"label": "host", "home": Path.home(), "email": "", "eligible": None, "host": True},
            {"label": "giovani", "home": Path(self.tmp.name) / "g", "email": "g@example.com", "host": False},
        ]
        self.env = patch.dict(
            os.environ,
            {"ANTIGRAVITY_ACCOUNTS_FILE": str(self.registry), "ANTIGRAVITY_ACCOUNTS_DIR": str(Path(self.tmp.name))},
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _run_list(self, probe_return, fast=False):
        out = io.StringIO()
        with patch("accounts.list_accounts", return_value=self.accounts), patch(
            "accounts.probe", return_value=probe_return
        ), patch("sys.stdout", out):
            code = _cmd_list(fast=fast)
        return code, out.getvalue()

    def test_layout_has_expected_columns_and_active_marker(self):
        code, text = self._run_list((True, {"gemini_5h": 1.0, "gemini_weekly": 0.86}, "", {}))
        self.assertEqual(code, 0)
        header = text.splitlines()[0]
        for column in ("ACCOUNT", "EMAIL", "ELIGIBLE", "GEMINI 5h/wk", "CLAUDE+GPT 5h/wk", "RESET", "HOME"):
            self.assertIn(column, header)
        self.assertNotIn("WARNING", text)

    def test_warning_only_for_ineligible(self):
        def probe_for(home):
            label = next(a["label"] for a in self.accounts if str(a["home"]) == str(home))
            if label == "giovani":
                return False, {}, "not eligible for Antigravity", {}
            return True, {"gemini_5h": 1.0, "gemini_weekly": 1.0}, "", {}

        out = io.StringIO()
        with patch("accounts.list_accounts", return_value=self.accounts), patch(
            "accounts.probe", side_effect=probe_for
        ), patch("sys.stdout", out):
            code = _cmd_list(fast=False)
        text = out.getvalue()
        self.assertEqual(code, 0)
        warnings = [line for line in text.splitlines() if line.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("giovani", warnings[0])

    def test_fast_mode_skips_probe(self):
        with patch("accounts.list_accounts", return_value=self.accounts), patch("accounts.probe") as probe, patch(
            "sys.stdout", io.StringIO()
        ):
            _cmd_list(fast=True)
        probe.assert_not_called()


class UseModeIgniteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = Path(self.tmp.name) / "accounts.json"
        self.env = patch.dict(
            os.environ,
            {
                "ANTIGRAVITY_ACCOUNTS_FILE": str(self.registry),
                "ANTIGRAVITY_ACCOUNTS_DIR": str(Path(self.tmp.name) / "dir"),
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _capture(self, func, *args):
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = func(*args)
        return code, out.getvalue()

    def test_use_sets_and_clears_active(self):
        account = {"label": "giovani", "home": Path("/tmp/g"), "email": "g@example.com", "host": False}
        with patch("accounts.resolve_account", return_value=account):
            code, text = self._capture(_cmd_use, "giovani")
        self.assertEqual(code, 0)
        self.assertIn("giovani", text)
        self.assertEqual(accounts.load_registry()["active_account"], "giovani")

        code, text = self._capture(_cmd_use, "host")
        self.assertEqual(code, 0)
        self.assertNotIn("active_account", accounts.load_registry())

    def test_use_unknown_account(self):
        with patch("accounts.resolve_account", return_value=None):
            code, text = self._capture(_cmd_use, "nope")
        self.assertEqual(code, 2)
        self.assertIn("Unknown account", text)

    def test_mode_show_default_and_persist(self):
        code, text = self._capture(_cmd_mode, "")
        self.assertEqual(code, 0)
        self.assertIn("off", text)

        code, _ = self._capture(_cmd_mode, "quota")
        self.assertEqual(code, 0)
        self.assertEqual(accounts.load_registry()["rotation_mode"], "quota")

        code, text = self._capture(_cmd_mode, "")
        self.assertIn("quota", text)

        code, text = self._capture(_cmd_mode, "bogus")
        self.assertEqual(code, 2)
        self.assertIn("Invalid mode", text)

    def test_ignite_toggle_and_status(self):
        code, text = self._capture(_cmd_ignite, "off")
        self.assertEqual(code, 0)
        self.assertIs(accounts.load_registry()["quota_ignition"], False)

        code, text = self._capture(_cmd_ignite, "on")
        self.assertEqual(code, 0)
        self.assertIs(accounts.load_registry()["quota_ignition"], True)

        code, text = self._capture(_cmd_ignite, "status")
        self.assertEqual(code, 0)
        self.assertIn("on", text)

    def test_unknown_action_prints_usage(self):
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = antigravity_command(type("A", (), {})())
        self.assertEqual(code, 2)
        self.assertIn("Usage: hermes antigravity", out.getvalue())


class AddCommandTests(unittest.TestCase):
    def test_add_without_tty_refuses_and_points_to_the_real_command(self):
        out = io.StringIO()
        with patch("sys.stdin") as stdin, patch("sys.stdout", out), patch("cli.subprocess.call") as call:
            stdin.isatty.return_value = False
            code = _cmd_add("")
        self.assertEqual(code, 2)
        self.assertIn("interactive terminal", out.getvalue())
        self.assertIn("hermes auth add antigravity-subscription-directsdk", out.getvalue())
        call.assert_not_called()

    def test_add_forwards_label_to_auth_add(self):
        out = io.StringIO()
        with patch("sys.stdin") as stdin, patch("sys.stdout", out), patch(
            "cli.shutil.which", return_value="/usr/local/bin/hermes"
        ), patch("cli.subprocess.call", return_value=0) as call:
            stdin.isatty.return_value = True
            code = _cmd_add("work")
        self.assertEqual(code, 0)
        call.assert_called_once_with(
            ["/usr/local/bin/hermes", "auth", "add", "antigravity-subscription-directsdk", "--label", "work"]
        )

    def test_add_without_label_omits_the_flag(self):
        with patch("sys.stdin") as stdin, patch("sys.stdout", io.StringIO()), patch(
            "cli.shutil.which", return_value="/usr/local/bin/hermes"
        ), patch("cli.subprocess.call", return_value=0) as call:
            stdin.isatty.return_value = True
            code = _cmd_add("")
        self.assertEqual(code, 0)
        call.assert_called_once_with(
            ["/usr/local/bin/hermes", "auth", "add", "antigravity-subscription-directsdk"]
        )


if __name__ == "__main__":
    unittest.main()
