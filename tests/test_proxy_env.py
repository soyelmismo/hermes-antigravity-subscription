"""Tests for the opt-in ANTIGRAVITY_PROXY child-environment wiring.

build_child_env() must leave the child environment untouched when
ANTIGRAVITY_PROXY is unset, and route agy's outbound HTTP(S) through the proxy
(while keeping loopback traffic direct and HOME isolation intact) when set.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from process import build_child_env

PROXY = "socks5h://127.0.0.1:1080"
_PROXY_KEYS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
)
_LOOPBACK = ["localhost", "127.0.0.1", "::1"]


def _base_env(**extra):
    """Minimal parent environment: no proxy, no Windows home variables."""
    env = {"PATH": os.environ.get("PATH", "")}
    env.update(extra)
    return env


class ProxyEnvTests(unittest.TestCase):
    def _build(self, parent_env, home):
        with patch.dict(os.environ, parent_env, clear=True):
            return build_child_env(home)

    def test_unset_leaves_env_unchanged(self):
        parent = _base_env(HTTPS_PROXY="http://inherited:3128", FOO="bar")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        expected = dict(parent, HOME=tmp, USERPROFILE=tmp, AGY_CLI_DISABLE_AUTO_UPDATE="true")
        self.assertEqual(env, expected)
        self.assertEqual(env["HTTPS_PROXY"], "http://inherited:3128")
        self.assertNotIn("NO_PROXY", env)
        self.assertNotIn("no_proxy", env)

    def test_blank_value_is_treated_as_unset(self):
        parent = _base_env(ANTIGRAVITY_PROXY="   ")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env, dict(parent, HOME=tmp, USERPROFILE=tmp, AGY_CLI_DISABLE_AUTO_UPDATE="true"))

    def test_set_populates_all_proxy_variables(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY)
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        for key in _PROXY_KEYS:
            self.assertEqual(env[key], PROXY, key)

    def test_set_overrides_inherited_proxy(self):
        parent = _base_env(
            ANTIGRAVITY_PROXY=PROXY,
            HTTPS_PROXY="http://inherited:3128",
            ALL_PROXY="socks5://corp-proxy:1080",
            all_proxy="socks5://corp-proxy:1080",
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        for key in _PROXY_KEYS:
            self.assertEqual(env[key], PROXY, key)

    def test_no_proxy_defaults_to_loopback(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY)
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env["NO_PROXY"].split(","), _LOOPBACK)
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_existing_no_proxy_is_preserved_and_extended(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY, NO_PROXY="corp.local")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env["NO_PROXY"], "corp.local,localhost,127.0.0.1,::1")
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_lowercase_no_proxy_is_preserved(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY, no_proxy="corp.local")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env["NO_PROXY"], "corp.local,localhost,127.0.0.1,::1")

    def test_both_no_proxy_casings_are_merged(self):
        parent = _base_env(
            ANTIGRAVITY_PROXY=PROXY,
            NO_PROXY="corp.local",
            no_proxy="internal.local,corp.local",
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(
            set(env["NO_PROXY"].split(",")),
            {"corp.local", "internal.local", *_LOOPBACK},
        )
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])

    def test_no_proxy_dedup_is_case_insensitive(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY, NO_PROXY="Corp.Local,LOCALHOST")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(
            env["NO_PROXY"].split(","),
            ["Corp.Local", "LOCALHOST", "127.0.0.1", "::1"],
        )

    def test_no_duplicate_no_proxy_entries(self):
        parent = _base_env(
            ANTIGRAVITY_PROXY=PROXY,
            NO_PROXY="corp.local, localhost ,127.0.0.1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        entries = env["NO_PROXY"].split(",")
        self.assertEqual(entries, ["corp.local", "localhost", "127.0.0.1", "::1"])
        self.assertEqual(len(entries), len(set(entries)))

    def test_home_isolation_still_applies_with_proxy(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY, HOME="/real/home")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env["HOME"], tmp)
        self.assertEqual(env["USERPROFILE"], tmp)

    def test_homepath_is_redirected_with_proxy(self):
        parent = _base_env(ANTIGRAVITY_PROXY=PROXY, HOMEPATH="\\Users\\real")
        with tempfile.TemporaryDirectory() as tmp:
            env = self._build(parent, tmp)
        self.assertEqual(env["HOMEPATH"], tmp)


if __name__ == "__main__":
    unittest.main()
