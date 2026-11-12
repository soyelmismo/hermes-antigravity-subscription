"""OAuth token sync and auth-contention detection for the agy client.

Multiple agy processes share one Google account; whoever refreshes the
shared token last wins, and a worker spawned against a stale copy dies with
"You are not logged into Antigravity" (observed verbatim in isolated-home
cli logs, 2026-11-11, while another Hermes session ran antigravity turns on
the same account). Pins:

- sync_isolated_token: the isolated token is a symlink to the real file
  (relinked when stale) or a byte-identical copy — never left behind stale;
- the client's single auth-contention retry fires ONLY on the exact
  signature: an empty no-result completion PLUS the token-source error in
  the worker's stderr tail.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import process as agy_process  # noqa: E402
from client import _empty_no_result, _stderr_shows_auth_contention  # noqa: E402


class SyncIsolatedTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.real_dir = Path(self.tmp.name) / "real" / ".gemini" / "antigravity-cli"
        self.real_dir.mkdir(parents=True)
        self.real_token = self.real_dir / "antigravity-oauth-token"
        self.real_token.write_text('{"token": "v1"}')

    def _isolated(self) -> Path:
        home = Path(self.tmp.name) / "iso"
        (home / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)
        return home

    def test_symlink_to_real_is_noop(self):
        home = self._isolated()
        token = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        os.symlink(self.real_token, token)
        with patch.object(agy_process, "resolve_real_token_path", return_value=self.real_token):
            self.assertTrue(agy_process.sync_isolated_token(home))
        self.assertTrue(token.is_symlink())
        self.assertEqual(Path(os.readlink(token)).resolve(), self.real_token.resolve())

    def test_stale_symlink_relinked(self):
        home = self._isolated()
        token = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        stale = self.real_dir / "stale-token"
        stale.write_text('{"token": "old"}')
        os.symlink(stale, token)
        with patch.object(agy_process, "resolve_real_token_path", return_value=self.real_token):
            self.assertTrue(agy_process.sync_isolated_token(home))
        self.assertEqual(token.read_text(), '{"token": "v1"}')

    def test_stale_copy_updated(self):
        home = self._isolated()
        token = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        token.write_text('{"token": "stale-copy"}')
        self.real_token.write_text('{"token": "v2"}')
        with patch.object(agy_process, "resolve_real_token_path", return_value=self.real_token):
            self.assertTrue(agy_process.sync_isolated_token(home))
        self.assertEqual(token.read_text(), '{"token": "v2"}')

    def test_identical_copy_not_rewritten(self):
        home = self._isolated()
        token = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        token.write_text('{"token": "v1"}')
        inode_before = token.stat().st_ino
        with patch.object(agy_process, "resolve_real_token_path", return_value=self.real_token):
            self.assertTrue(agy_process.sync_isolated_token(home))
        self.assertEqual(token.stat().st_ino, inode_before)

    def test_missing_isolated_dir_created(self):
        home = Path(self.tmp.name) / "fresh-home"
        with patch.object(agy_process, "resolve_real_token_path", return_value=self.real_token):
            self.assertTrue(agy_process.sync_isolated_token(home))
        linked = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        self.assertEqual(linked.read_text(), '{"token": "v1"}')

    def test_no_real_token_false(self):
        home = self._isolated()
        with patch.object(agy_process, "resolve_real_token_path", return_value=None):
            self.assertFalse(agy_process.sync_isolated_token(home))


class _FakeDrainer:
    def __init__(self, tail: str):
        self._tail = tail

    def get_tail(self, timeout: float = 1.0) -> str:
        return self._tail


def _completion(content=None, finish_reason=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ]
    )


class EmptyNoResultTests(unittest.TestCase):
    def test_empty_completion_detected(self):
        self.assertTrue(_empty_no_result(_completion()))

    def test_content_present_not_empty(self):
        self.assertFalse(_empty_no_result(_completion(content="hola")))

    def test_tool_calls_present_not_empty(self):
        self.assertFalse(_empty_no_result(_completion(tool_calls=[object()])))

    def test_finish_reason_present_not_empty(self):
        self.assertFalse(_empty_no_result(_completion(finish_reason="stop")))

    def test_garbage_safe_false(self):
        self.assertFalse(_empty_no_result(None))
        self.assertFalse(_empty_no_result(object()))


class AuthContentionDetectionTests(unittest.TestCase):
    def test_signature_detected(self):
        proc = SimpleNamespace(
            _stderr_drainer=_FakeDrainer(
                "E errorreport error getting token source: You are not logged into Antigravity."
            )
        )
        self.assertTrue(_stderr_shows_auth_contention(proc))

    def test_benign_stderr_not_flagged(self):
        proc = SimpleNamespace(_stderr_drainer=_FakeDrainer("I printmode Print mode: slash command expansion disabled"))
        self.assertFalse(_stderr_shows_auth_contention(proc))

    def test_no_drainer_safe_false(self):
        self.assertFalse(_stderr_shows_auth_contention(SimpleNamespace()))
        self.assertFalse(_stderr_shows_auth_contention(None))


if __name__ == "__main__":
    unittest.main()
