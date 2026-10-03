"""Tests for the Antigravity companion CLI plugin (no agy calls, no network)."""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import accounts  # noqa: E402
from cli import _cmd_list, _cmd_usage, antigravity_command  # noqa: E402


def make_id_token(email: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


USAGE_JSON = {
    "status": "SUCCESS",
    "num_turns": 0,
    "command": {
        "name": "usage",
        "data": {
            "groups": [
                {
                    "name": "Gemini Models",
                    "buckets": [
                        {"id": "gemini-5h", "window": "5h", "remaining_fraction": 0.5},
                        {"id": "gemini-weekly", "window": "weekly", "remaining_fraction": 0.8},
                    ],
                },
                {
                    "name": "Claude and GPT models",
                    "buckets": [
                        {"id": "3p-5h", "window": "5h", "remaining_fraction": 1.0},
                        {"id": "3p-weekly", "window": "weekly", "remaining_fraction": 0.25},
                    ],
                },
            ]
        },
    },
}


class TokenTests(unittest.TestCase):
    def test_email_from_id_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "antigravity-oauth-token"
            token_file.write_text(json.dumps({"id_token": make_id_token("someone@example.com"), "auth_method": "consumer"}))
            self.assertEqual(accounts.email_from_token(token_file), "someone@example.com")

    def test_email_missing_or_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "antigravity-oauth-token"
            broken.write_text("{not json")
            self.assertEqual(accounts.email_from_token(broken), "")
        self.assertEqual(accounts.email_from_token(None), "")

    def test_email_from_nested_token_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "jetski-standalone-oauth-token"
            token_file.write_text(json.dumps({"token": {"id_token": make_id_token("nested@example.com")}}))
            self.assertEqual(accounts.email_from_token(token_file), "nested@example.com")


class ProbeTests(unittest.TestCase):
    def _run(self, stdout: str = "", stderr: str = "", code: int = 0):
        return patch(
            "accounts.subprocess.run",
            return_value=type("R", (), {"stdout": stdout, "stderr": stderr, "returncode": code})(),
        )

    def test_probe_eligible_parses_windows(self):
        with self._run(stdout=json.dumps(USAGE_JSON)):
            eligible, windows, note, resets = accounts.probe(Path.home())
        self.assertTrue(eligible)
        self.assertEqual(note, "")
        self.assertAlmostEqual(windows["gemini_5h"], 0.5)
        self.assertAlmostEqual(windows["claude_gpt_weekly"], 0.25)

    def test_probe_ineligible(self):
        with self._run(stdout="error: Eligibility check failed: Your current account is not eligible for Antigravity."):
            eligible, windows, note, resets = accounts.probe(Path.home())
        self.assertFalse(eligible)
        self.assertIn("not eligible", note)

    def test_probe_not_signed_in(self):
        with self._run(stdout="Please sign in to view available models."):
            eligible, _, note, _ = accounts.probe(Path.home())
        self.assertIsNone(eligible)
        self.assertEqual(note, "not signed in")

    def test_probe_timeout(self):
        with patch("accounts.subprocess.run", side_effect=accounts.subprocess.TimeoutExpired(cmd="agy", timeout=1)):
            eligible, _, note, _ = accounts.probe(Path.home())
        self.assertIsNone(eligible)
        self.assertEqual(note, "timeout")


class AccountListingTests(unittest.TestCase):
    def test_host_is_first_and_registry_entries_follow(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "acct"
            (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
            (home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token").write_text(
                json.dumps({"id_token": make_id_token("reg@example.com")})
            )
            registry = Path(tmp) / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "accounts": [
                            {"label": "reg", "home_dir": str(home), "enabled": True, "eligible": True}
                        ],
                        "active_account": "reg",
                    }
                )
            )
            with patch.dict(
                "os.environ",
                {"ANTIGRAVITY_ACCOUNTS_FILE": str(registry), "ANTIGRAVITY_ACCOUNTS_DIR": str(Path(tmp) / "none")},
            ):
                entries = accounts.list_accounts()
        self.assertEqual(entries[0]["label"], "host")
        self.assertTrue(entries[0]["host"])
        self.assertEqual(entries[1]["label"], "reg")
        self.assertEqual(entries[1]["email"], "reg@example.com")


class CommandTests(unittest.TestCase):
    def _list_output(self, probe_result, accounts_list, fast=False, active=""):
        out = io.StringIO()
        if len(probe_result) == 3:
            probe_result = probe_result + ({},)
        with patch("accounts.list_accounts", return_value=accounts_list), patch(
            "accounts.probe", return_value=probe_result
        ), patch("accounts.active_store_name", return_value=active), patch("sys.stdout", out):
            code = _cmd_list(fast=fast)
        return code, out.getvalue()

    def test_list_warns_only_for_ineligible(self):
        accounts_list = [
            {"label": "host", "home": Path.home(), "email": "", "eligible": None, "host": True},
            {"label": "bad", "home": Path("/tmp/bad"), "email": "bad@example.com", "eligible": False, "host": False},
        ]
        code, text = self._list_output((False, {}, "not eligible for Antigravity"), accounts_list)
        self.assertEqual(code, 0)
        warnings = [line for line in text.splitlines() if line.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("bad", warnings[0])

    def test_list_is_quiet_when_all_eligible(self):
        accounts_list = [
            {"label": "host", "home": Path.home(), "email": "", "eligible": None, "host": True},
        ]
        code, text = self._list_output((True, {"gemini_5h": 1.0, "gemini_weekly": 1.0}, ""), accounts_list)
        self.assertEqual(code, 0)
        self.assertNotIn("WARNING", text)
        self.assertIn("100%", text)

    def test_usage_command_reports_ineligible_with_warning(self):
        account = {"label": "bad", "home": Path("/tmp/bad"), "email": "bad@example.com", "host": False}
        out = io.StringIO()
        with patch("accounts.resolve_account", return_value=account), patch(
            "accounts.probe", return_value=(False, {}, "not eligible for Antigravity", {})
        ), patch("sys.stdout", out):
            code = _cmd_usage("bad")
        self.assertEqual(code, 1)
        self.assertIn("WARNING", out.getvalue())

    def test_unknown_action_returns_usage(self):
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = antigravity_command(type("A", (), {})())
        self.assertEqual(code, 2)
        self.assertIn("Usage: hermes antigravity", out.getvalue())

    def test_run_strips_leading_separator_and_forwards_args(self):
        account = {"label": "acc", "home": Path("/tmp/acc"), "email": "", "host": False}
        with patch("accounts.resolve_account", return_value=account), patch(
            "accounts.run_agy", return_value=0
        ) as run:
            args = type("A", (), {"antigravity_action": "run", "label": "acc", "agy_args": ["--", "-p", "hi"]})()
            code = antigravity_command(args)
        self.assertEqual(code, 0)
        run.assert_called_once_with(Path("/tmp/acc"), ["-p", "hi"])

    def test_run_unknown_account(self):
        out = io.StringIO()
        with patch("accounts.resolve_account", return_value=None), patch("sys.stdout", out):
            args = type("A", (), {"antigravity_action": "run", "label": "nope", "agy_args": []})()
            code = antigravity_command(args)
        self.assertEqual(code, 2)
        self.assertIn("Unknown account", out.getvalue())


if __name__ == "__main__":
    unittest.main()
