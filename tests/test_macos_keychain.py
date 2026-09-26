"""Regression tests for the macOS keychain half of issue #1.

On macOS agy 1.2 keeps its session in the login keychain (go-keyring's darwin
backend: a generic password with service ``gemini`` and account
``antigravity``) instead of a token file. Two things follow:

* ``is_authenticated()`` has to notice that item without reading it, which
  ``security find-generic-password`` without ``-g``/``-w`` does: it prints the
  item's attributes to stdout and never the secret.
* The child agy runs under an isolated HOME, and the keychain search list is
  resolved through ``$HOME/Library/Keychains``. Without that directory the
  child answers "Authentication required" even for a logged-in user, so
  ``setup_isolated_home()`` links it in.

Ground truth (agy 1.2.11, macOS): ``security find-generic-password -s gemini
-a antigravity`` exits 0 with ``"svce"<blob>="gemini"`` and
``"acct"<blob>="antigravity"`` on stdout; a miss exits 44; agy under a bare
temporary HOME asks for login, and the same HOME with ``Library/Keychains``
linked to the real directory answers normally.

No host state is consulted: the platform is faked inside the process module,
the environment is scrubbed, ``subprocess.run`` is mocked, and HOME is a
temporary directory.
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import process
from process import is_authenticated, setup_isolated_home

_EXPECTED_ARGV = ["/usr/bin/security", "find-generic-password", "-s", "gemini", "-a", "antigravity"]

# What `security find-generic-password -s gemini -a antigravity` prints on a
# hit (sanitized; the real output lists more attributes, never the secret).
_SECURITY_STDOUT_HIT = (
    b'keychain: "/Users/someone/Library/Keychains/login.keychain-db"\n'
    b"version: 512\n"
    b'class: "genp"\n'
    b"attributes:\n"
    b'    "acct"<blob>="antigravity"\n'
    b'    "svce"<blob>="gemini"\n'
)


def _security_result(stdout=b"", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=None, returncode=returncode)


@contextmanager
def _macos(home=None, config_dir=None):
    """darwin inside the process module, scrubbed env, no token file anywhere."""
    env = {k: v for k, v in os.environ.items() if k != "ANTIGRAVITY_CONFIG_DIR"}
    for var in ("AGY_CLI_PATH", "ANTIGRAVITY_COMMAND", "ANTIGRAVITY_CLI_PATH"):
        env.pop(var, None)
    with contextlib.ExitStack() as stack:
        if home is None:
            home = stack.enter_context(tempfile.TemporaryDirectory())
        env["HOME"] = str(home)
        # Path.home() reads USERPROFILE on Windows runners, where darwin is faked.
        env["USERPROFILE"] = str(home)
        if config_dir is not None:
            env["ANTIGRAVITY_CONFIG_DIR"] = str(config_dir)
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        stack.enter_context(patch("process._is_existing_file", return_value=False))
        mock_os = stack.enter_context(patch("process.os", wraps=os))
        mock_sys = stack.enter_context(patch("process.sys", wraps=sys))
        mock_os.name = "posix"
        mock_sys.platform = "darwin"
        yield Path(home)


class KeychainProbeTests(unittest.TestCase):
    """``is_authenticated()`` on macOS with no token file."""

    def test_keychain_item_authenticates(self):
        with _macos(), patch("process.subprocess.run", return_value=_security_result(_SECURITY_STDOUT_HIT)) as run:
            self.assertIs(is_authenticated(), True)
            # Exactly one spawn, and it is the security probe: no secret-tool or
            # interpreter call may precede it.
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], _EXPECTED_ARGV)

    def test_darwin_never_reaches_the_linux_probe(self):
        for returncode, stdout in ((0, _SECURITY_STDOUT_HIT), (44, b"")):
            with self.subTest(returncode=returncode), _macos(), patch(
                "process._linux_keyring_present", side_effect=AssertionError("Linux probe ran on darwin")
            ), patch("process.subprocess.run", return_value=_security_result(stdout, returncode=returncode)) as run:
                self.assertIs(is_authenticated(), returncode == 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], _EXPECTED_ARGV)

    def test_probe_never_asks_for_the_secret(self):
        # -g prints the password to stderr and -w prints it alone to stdout;
        # neither may ever be part of the argv.
        with _macos(), patch("process.subprocess.run", return_value=_security_result(_SECURITY_STDOUT_HIT)) as run:
            is_authenticated()
            argv = run.call_args.args[0]
            self.assertNotIn("-g", argv)
            self.assertNotIn("-w", argv)
            self.assertIs(run.call_args.kwargs.get("stderr"), subprocess.DEVNULL)
            self.assertIn("timeout", run.call_args.kwargs)

    def test_item_not_found_is_unauthenticated(self):
        # rc 44 is errSecItemNotFound: "The specified item could not be found".
        with _macos(), patch("process.subprocess.run", return_value=_security_result(returncode=44)):
            self.assertIs(is_authenticated(), False)

    def test_zero_rc_without_the_exact_pair_is_unauthenticated(self):
        other = _SECURITY_STDOUT_HIT.replace(b'"svce"<blob>="gemini"', b'"svce"<blob>="gemini-cli"')
        for stdout in (b"", other):
            with self.subTest(stdout=stdout), _macos(), patch(
                "process.subprocess.run", return_value=_security_result(stdout)
            ):
                self.assertIs(is_authenticated(), False)

    def test_spawn_failures_fail_closed(self):
        for error in (FileNotFoundError("security"), subprocess.TimeoutExpired(_EXPECTED_ARGV, 5.0)):
            with self.subTest(error=type(error).__name__), _macos(), patch(
                "process.subprocess.run", side_effect=error
            ):
                self.assertIs(is_authenticated(), False)

    def test_explicit_config_dir_blocks_the_probe(self):
        with tempfile.TemporaryDirectory() as config_dir, _macos(config_dir=config_dir), patch(
            "process.subprocess.run", return_value=_security_result(_SECURITY_STDOUT_HIT)
        ) as run:
            self.assertIs(is_authenticated(), False)
            run.assert_not_called()

    def test_argv_and_markers_derive_from_the_shared_constants(self):
        self.assertEqual(process._SECURITY_PROBE_ARGV, _EXPECTED_ARGV)
        self.assertEqual(
            process._SECURITY_HIT_MARKERS,
            (
                b'"svce"<blob>="' + process._KEYRING_SERVICE.encode() + b'"',
                b'"acct"<blob>="' + process._KEYRING_USERNAME.encode() + b'"',
            ),
        )


class IsolatedHomeKeychainTests(unittest.TestCase):
    """``setup_isolated_home()`` exposes the keychains to the child agy on macOS."""

    def _home_with_keychains(self, stack):
        home = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        (home / "Library" / "Keychains").mkdir(parents=True)
        return home

    def test_links_the_real_keychains_directory(self):
        with contextlib.ExitStack() as stack:
            home = self._home_with_keychains(stack)
            cwd = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(_macos(home=home))
            isolated_home, _ = setup_isolated_home(cwd)
            link = isolated_home / "Library" / "Keychains"
            self.assertTrue(os.path.islink(link))
            self.assertEqual(link.resolve(), (home / "Library" / "Keychains").resolve())

    def test_reused_cwd_relinks_a_stale_link(self):
        with contextlib.ExitStack() as stack:
            home = self._home_with_keychains(stack)
            elsewhere = stack.enter_context(tempfile.TemporaryDirectory())
            cwd = stack.enter_context(tempfile.TemporaryDirectory())
            stale = Path(cwd) / "home" / "Library" / "Keychains"
            stale.parent.mkdir(parents=True)
            os.symlink(elsewhere, stale)
            stack.enter_context(_macos(home=home))
            setup_isolated_home(cwd)
            setup_isolated_home(cwd)  # idempotent on a second run
            self.assertEqual(stale.resolve(), (home / "Library" / "Keychains").resolve())

    def test_a_real_directory_is_never_replaced(self):
        with contextlib.ExitStack() as stack:
            home = self._home_with_keychains(stack)
            cwd = stack.enter_context(tempfile.TemporaryDirectory())
            existing = Path(cwd) / "home" / "Library" / "Keychains"
            existing.mkdir(parents=True)
            (existing / "keep").write_text("user data")
            stack.enter_context(_macos(home=home))
            setup_isolated_home(cwd)
            self.assertFalse(os.path.islink(existing))
            self.assertEqual((existing / "keep").read_text(), "user data")

    def test_skipped_off_macos_without_keychains_or_with_explicit_config_dir(self):
        with contextlib.ExitStack() as stack:
            home = self._home_with_keychains(stack)
            bare_home = stack.enter_context(tempfile.TemporaryDirectory())
            config_dir = stack.enter_context(tempfile.TemporaryDirectory())
            cases = (
                ("linux", home, None),
                ("darwin", bare_home, None),
                ("darwin", home, config_dir),
            )
            for platform_name, case_home, case_config in cases:
                with self.subTest(platform=platform_name, home=case_home, config=case_config):
                    cwd = stack.enter_context(tempfile.TemporaryDirectory())
                    with _macos(home=case_home, config_dir=case_config), patch("process.sys.platform", platform_name):
                        isolated_home, _ = setup_isolated_home(cwd)
                    self.assertFalse(os.path.lexists(isolated_home / "Library" / "Keychains"))


if __name__ == "__main__":
    unittest.main()
