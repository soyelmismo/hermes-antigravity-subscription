"""Regression tests for dual OAuth-token filenames (issue #1).

agy1.2 renamed the fallback token file from ``antigravity-oauth-token`` to
``jetski-standalone-oauth-token``. process.resolve_real_token_path() must
accept both; setup_isolated_home() must preserve whichever basename was
selected. No production secret is ever parsed here — tests use temp files
with dummy content; only the copy2-fallback test reads bytes back to prove
the copy landed.
"""

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import process
from process import resolve_real_token_path, setup_isolated_home

LEGACY_NAME = "antigravity-oauth-token"
NEW_NAME = "jetski-standalone-oauth-token"
_TOKEN_BODY = "token-content-123456"


@contextmanager
def _scrubbed_env(**overrides):
    """Environ without ANTIGRAVITY_CONFIG_DIR leakage, plus overrides.

    Mirrors build_child_env(): pathlib's ntpath.expanduser() consults
    USERPROFILE then HOMEDRIVE/HOMEPATH — never HOME — so HOME alone does
    not steer Path.home() on Windows. Any HOME= override is therefore also
    applied to USERPROFILE, and HOMEDRIVE/HOMEPATH are dropped, so the temp
    dir wins on both platforms.
    """
    env = dict(os.environ)
    env.pop("ANTIGRAVITY_CONFIG_DIR", None)
    if "HOME" in overrides:
        overrides.setdefault("USERPROFILE", overrides["HOME"])
    env.pop("HOMEDRIVE", None)
    env.pop("HOMEPATH", None)
    env.update(overrides)
    with patch.dict(os.environ, env, clear=True):
        yield


def _write_token(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TOKEN_BODY, encoding="utf-8")
    return path


class AuthFileCompatibilityTests(unittest.TestCase):
    def test_legacy_filename_resolves_in_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = _write_token(Path(tmp) / ".gemini" / "antigravity-cli" / LEGACY_NAME)
            with _scrubbed_env(HOME=tmp):
                self.assertEqual(resolve_real_token_path(), token)

    def test_new_filename_resolves_in_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = _write_token(Path(tmp) / ".gemini" / "antigravity-cli" / NEW_NAME)
            with _scrubbed_env(HOME=tmp):
                self.assertEqual(resolve_real_token_path(), token)

    def test_explicit_override_supports_both_names(self):
        for name in (LEGACY_NAME, NEW_NAME):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as home:
                    token = _write_token(Path(config_dir) / name)
                    with _scrubbed_env(HOME=home, ANTIGRAVITY_CONFIG_DIR=config_dir):
                        self.assertEqual(resolve_real_token_path(), token)

    def test_legacy_wins_when_both_exist_in_home(self):
        # Documented tie-break: legacy-first order means an old token file
        # lingering next to the new one still wins.
        with tempfile.TemporaryDirectory() as tmp:
            legacy = _write_token(Path(tmp) / ".gemini" / "antigravity-cli" / LEGACY_NAME)
            _write_token(Path(tmp) / ".gemini" / "antigravity-cli" / NEW_NAME)
            with _scrubbed_env(HOME=tmp):
                self.assertEqual(resolve_real_token_path(), legacy)

    def test_legacy_wins_when_both_exist_in_override(self):
        with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as home:
            legacy = _write_token(Path(config_dir) / LEGACY_NAME)
            _write_token(Path(config_dir) / NEW_NAME)
            with _scrubbed_env(HOME=home, ANTIGRAVITY_CONFIG_DIR=config_dir):
                self.assertEqual(resolve_real_token_path(), legacy)

    def test_explicit_override_is_strict(self):
        # A token in HOME must not satisfy auth when the override points
        # at an empty directory.
        with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as home:
            _write_token(Path(home) / ".gemini" / "antigravity-cli" / LEGACY_NAME)
            with _scrubbed_env(HOME=home, ANTIGRAVITY_CONFIG_DIR=config_dir):
                self.assertIsNone(resolve_real_token_path())

    def test_home_new_beats_root_legacy_fallback(self):
        # Whatever exists under /root must never outrank the user's HOME
        # token, whichever filename each side uses.
        with tempfile.TemporaryDirectory() as home:
            home_token = _write_token(Path(home) / ".gemini" / "antigravity-cli" / NEW_NAME)
            real_probe = process._is_existing_file

            def fake_probe(path):
                if str(path).startswith("/root/"):
                    return True  # pretend a stale legacy token lingers in /root
                return real_probe(path)

            with _scrubbed_env(HOME=home), patch(
                "process._is_existing_file", side_effect=fake_probe
            ):
                self.assertEqual(resolve_real_token_path(), home_token)

    def test_isolated_home_preserves_legacy_basename(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as src_dir:
            src = _write_token(Path(src_dir) / LEGACY_NAME)
            with patch("process.resolve_real_token_path", return_value=src):
                _, gemini_dir = setup_isolated_home(tmp)
                self.assertTrue((gemini_dir / LEGACY_NAME).exists())
                self.assertFalse((gemini_dir / NEW_NAME).exists())

    def test_isolated_home_preserves_new_basename(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as src_dir:
            src = _write_token(Path(src_dir) / NEW_NAME)
            with patch("process.resolve_real_token_path", return_value=src):
                _, gemini_dir = setup_isolated_home(tmp)
                self.assertTrue((gemini_dir / NEW_NAME).exists())
                self.assertFalse((gemini_dir / LEGACY_NAME).exists())

    def test_isolated_home_link_fallback_chain_keeps_new_basename(self):
        # Symlink -> hardlink -> copy2 fallback must all land on the same
        # (new) basename.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as src_dir:
            src = _write_token(Path(src_dir) / NEW_NAME)
            with patch("process.resolve_real_token_path", return_value=src):
                with patch("process.os.symlink", side_effect=OSError("privilege not held")), patch(
                    "process.os.link", side_effect=OSError("cross-device link")
                ):
                    _, gemini_dir = setup_isolated_home(tmp)
                    dest = gemini_dir / NEW_NAME
                    self.assertTrue(dest.is_file())
                    self.assertEqual(dest.read_text(encoding="utf-8"), _TOKEN_BODY)


if __name__ == "__main__":
    unittest.main()
