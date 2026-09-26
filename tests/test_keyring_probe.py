"""Regression tests for the Linux keyring probe (issue #1).

On Linux with a D-Bus session `agy` stores its session credential in the
freedesktop Secret Service rather than in a token file, so
``process._linux_keyring_present()`` — wired into ``is_authenticated()`` after
``resolve_real_token_path()`` returns None — has to notice that credential
without ever receiving it. Headless systems without a session bus make agy skip
the keyring entirely, so the probe must simply degrade to False there and let
the token-file scan decide.

The attribute ground truth is the reporter's sanitized live dump from an
active agy session (issue #1 comment, 2026-09-26): service ``gemini``,
username ``antigravity``, label "Password for 'antigravity' on 'gemini'" — the
zalando/go-keyring convention, mirroring the Windows target
``gemini:antigravity`` from PR #2.

libsecret's secret-tool prints the matched credential itself on STDOUT
(verified against tool/secret-tool.c: search sets SECRET_SEARCH_LOAD_SECRETS
and prints ``secret = ...`` there, attributes on stderr), so the primary probe
must run with stdout closed. These tests pin that wiring as hard as the verdict
itself: a probe that captured the credential would pass every other assertion.

No host state is consulted: the OS and platform names are faked only inside
the process module (wraps the real ``os``/``sys`` so pathlib's own checks keep
working), the environment is scrubbed, ``shutil.which`` / ``os.path.exists`` /
``os.path.realpath`` / ``subprocess.run`` are all mocked, and the fallback
script is executed in-process against a stubbed ``secretstorage`` module.
"""

import ast
import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import process
from process import is_authenticated

_SECRET_TOOL = "secret-tool"
_PYTHON3 = "python3"

# The exact argv the primary mechanism must run, kept in sync with
# process._SECRET_TOOL_PROBE_ARGV by the constant-coherence tests below.
_EXPECTED_SECRET_TOOL_ARGV = ["secret-tool", "search", "--all", "service", "gemini", "username", "antigravity"]

# What secret-tool really emits on STDERR for agy's item: one
# `attribute.<name> = <value>` line per stored attribute (xdg:schema goes to
# stdout instead, so it is deliberately absent here). Its STDOUT would carry
# `[<object path>]`, `label = ...`, `secret = <credential>`, `created = ...`,
# `modified = ...` and `schema = ...` — output the verdict must never see.
# Every line is built the way libsecret prints it (g_printerr
# "attribute.%s = %s\n"), so the fixture cannot drift from the code's markers.
def _attribute_line(name, value):
    return b"attribute." + name.encode() + b" = " + value.encode() + b"\n"


_SECRET_TOOL_STDERR_HIT = (
    _attribute_line("service", process._KEYRING_SERVICE)
    + _attribute_line("username", process._KEYRING_USERNAME)
    + _attribute_line("x-other", "whatever")
)
_SECRET_TOOL_STDOUT_SECRET = (
    b"[org/freedesktop/secrets/collection/login/1]\n"
    b"label = Password for 'antigravity' on 'gemini'\n"
    b"secret = ya29.a0AfB...OAUTH-TOKEN\n"
    b"created = 2026-09-01 10:00:00\n"
)
_EXIT_ZERO = SimpleNamespace(stdout=b"", returncode=0)
_EXIT_THREE = SimpleNamespace(stdout=b"", stderr=b"", returncode=3)
_EXIT_TWO = SimpleNamespace(stdout=b"", stderr=b"", returncode=2)
# CPython's own code for an uncaught exception (a SyntaxError in an ancient
# interpreter, say), raised before the probe script's try/except ever runs.
_EXIT_BROKEN_INTERPRETER = SimpleNamespace(stdout=b"", stderr=b"", returncode=1)

# Deterministic stand-in for the cascade's bare "python3" entry.
_VENV_PYTHON = "/opt/hermes-venv/bin/python3"


def _secret_tool_result(stderr=b"", returncode=0, stdout=b""):
    """A subprocess result shaped like secret-tool's: metadata on stderr."""
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _run_side_effect(secret_tool=None, scripted=None):
    """Fake subprocess.run answering per probed command.

    ``scripted`` may be a list of results, one per interpreter in the cascade
    (consumed in spawn order), or a single result used for every scripted
    call. Either entry may be an exception instance to raise
    (FileNotFoundError, TimeoutExpired, ...).
    """

    def _run(argv, *args, **kwargs):
        if argv[:1] == [_SECRET_TOOL]:
            if isinstance(secret_tool, BaseException):
                raise secret_tool
            return secret_tool
        results = scripted if isinstance(scripted, list) else [scripted]
        result = results[0] if len(results) == 1 else results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    return _run


@contextmanager
def _probe_env(secret_tool="/usr/bin/secret-tool", bare_python=_VENV_PYTHON, executable=(), realpath_map=None):
    """Control every external dependency of the probe, host-independently.

    ``secret_tool=None`` hides the CLI from PATH; ``bare_python`` is what the
    bare ``python3`` cascade entry resolves to (None = not on PATH);
    ``executable`` names the absolute cascade entries that are executable here
    (``os.access(path, X_OK)``). ``realpath_map`` pins ``os.path.realpath`` —
    unknown paths pass through unchanged — so no test consults host symlinks:
    the real ``/usr/local/bin/python3`` and friends are never stat()ed.
    """
    realpath_map = realpath_map or {}

    def which(name):
        if name == _SECRET_TOOL:
            return secret_tool
        if name == _PYTHON3:
            return bare_python
        return None

    # ``os.X_OK`` as seen from inside the wrapped-mock process module is a Mock
    # rather than the real int, so the mode cannot be compared; the recorded
    # (path, mode) pairs are yielded so a test can still pin it.
    access_calls = []

    def access(path, mode):
        access_calls.append((path, mode))
        return path in executable

    with patch("process.shutil.which", side_effect=which), patch("process.os.access", side_effect=access), patch(
        "process.os.path.realpath", side_effect=lambda path: realpath_map.get(path, path)
    ):
        yield access_calls


@contextmanager
def _no_token_file(platform="linux", os_name="posix"):
    """Any platform, no token file anywhere, host environment scrubbed.

    ``_is_existing_file`` is pinned so neither the HOME candidates nor the
    /root fallback can be satisfied by a real token on the host, and the
    OS/platform names are faked only inside the process module (wraps the real
    os/sys) so pathlib's own checks keep working.
    """
    env = {k: v for k, v in os.environ.items() if k != "ANTIGRAVITY_CONFIG_DIR"}
    for var in ("AGY_CLI_PATH", "ANTIGRAVITY_COMMAND", "ANTIGRAVITY_CLI_PATH"):
        env.pop(var, None)
    with contextlib.ExitStack() as stack:
        home = stack.enter_context(tempfile.TemporaryDirectory())
        env["HOME"] = home
        env["USERPROFILE"] = home
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        stack.enter_context(patch("process._is_existing_file", return_value=False))
        mock_os = stack.enter_context(patch("process.os", wraps=os))
        mock_sys = stack.enter_context(patch("process.sys", wraps=sys))
        mock_os.name = os_name
        mock_sys.platform = platform
        yield


class SecretToolProbeTests(unittest.TestCase):
    """``secret-tool search`` — the primary mechanism, including stream safety."""

    def test_exact_attribute_pair_authenticates(self):
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(_SECRET_TOOL_STDERR_HIT)
        ) as run:
            self.assertIs(is_authenticated(), True)
            self.assertEqual(run.call_args.args[0], _EXPECTED_SECRET_TOOL_ARGV)
            self.assertEqual(run.call_args.args[0], process._SECRET_TOOL_PROBE_ARGV)
            self._assert_secret_tool_call(run.call_args.kwargs)

    def test_stdout_is_never_captured(self):
        # libsecret prints the credential on stdout, so the wiring must close
        # it. A stdout-matching probe would satisfy this fixture, which is how
        # the regression is caught: the verdict may not depend on stdout.
        stdout = _SECRET_TOOL_STDOUT_SECRET
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(_SECRET_TOOL_STDERR_HIT, stdout=stdout)
        ) as run:
            self.assertIs(is_authenticated(), True)
            self.assertIs(run.call_args.kwargs.get("stdout"), subprocess.DEVNULL)
        # Nor is a credential on stdout needed for a negative verdict.
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(b"", stdout=stdout)
        ) as run:
            self.assertFalse(is_authenticated())
            self.assertIs(run.call_args.kwargs.get("stdout"), subprocess.DEVNULL)

    def test_no_match_is_definitive_false(self):
        # rc 0 with no attribute lines: secret-tool found nothing. That is a
        # verdict, so the scripted probe must NOT run.
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(b"")
        ) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], _EXPECTED_SECRET_TOOL_ARGV)

    def test_single_attribute_line_is_not_a_match(self):
        # AND semantics: one of the two attribute lines is not a hit, and the
        # probe answers itself instead of handing off.
        for stderr in (b"attribute.service = gemini\n", b"attribute.username = antigravity\n", b"service = gemini\n"):
            with self.subTest(stderr=stderr), _no_token_file(), _probe_env(), patch(
                "process.subprocess.run", return_value=_secret_tool_result(stderr)
            ) as run:
                self.assertFalse(is_authenticated())
                self.assertEqual(run.call_count, 1)

    def test_unanchored_service_prefix_cannot_masquerade_as_a_hit(self):
        # `attribute.service = gemini-cli` CONTAINS `attribute.service = gemini`,
        # so the hit markers carry their trailing newline exactly as libsecret
        # prints them (g_printerr "attribute.%s = %s\n"). With the anchor, this
        # line cannot satisfy the service marker.
        stderr = (
            _attribute_line("service", process._KEYRING_SERVICE + "-cli")
            + _attribute_line("username", process._KEYRING_USERNAME)
        )
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(stderr)
        ) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual(run.call_count, 1)

    def test_attribute_lines_survive_console_code_page_noise(self):
        # stderr is read as bytes: undecodable surrounding output never raises
        # and never spoils the containment check.
        stderr = b"User: \x81\x8d\x90\xff\r\n" + _SECRET_TOOL_STDERR_HIT + b"\xfe\xff\n"
        with _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(stderr)
        ) as run:
            self.assertIs(is_authenticated(), True)
            self._assert_secret_tool_call(run.call_args.kwargs)

    def test_could_not_answer_hands_off_to_the_scripted_probe(self):
        # Timeout, spawn failure and a tool-level error are all "no verdict":
        # the scripted probe runs and its answer governs (a wedged D-Bus session
        # fails every mechanism equally, so the fallback must still get a turn).
        for reason in (
            subprocess.TimeoutExpired([_SECRET_TOOL], 5),
            FileNotFoundError(),
            PermissionError("d-bus not available"),
            _secret_tool_result(b"secret-tool: could not connect to the Secret Service\n", returncode=1),
        ):
            for scripted, expected in ((_EXIT_ZERO, True), (_EXIT_THREE, False)):
                with self.subTest(reason=repr(reason), scripted_exit=scripted.returncode), _no_token_file(), _probe_env(
                    executable=()
                ), patch(
                    "process.subprocess.run", side_effect=_run_side_effect(secret_tool=reason, scripted=scripted)
                ) as run:
                    self.assertIs(is_authenticated(), expected)
                    self.assertEqual(run.call_count, 2)
                    self.assertEqual(run.call_args_list[0].args[0], _EXPECTED_SECRET_TOOL_ARGV)
                    self.assertEqual(run.call_args_list[1].args[0][:2], [_VENV_PYTHON, "-c"])

    def test_missing_secret_tool_falls_back_to_scripted_probe(self):
        # Two shapes of "secret-tool is not usable here": absent from PATH
        # (which() says no) and vanishing between which() and the spawn
        # (FileNotFoundError). Both must reach the scripted probe.
        for secret_tool, spawn_error in ((None, None), ("/usr/bin/secret-tool", FileNotFoundError())):
            with self.subTest(secret_tool=secret_tool, spawn_error=repr(spawn_error)), _no_token_file(), _probe_env(
                secret_tool=secret_tool, executable=()
            ), patch(
                "process.subprocess.run",
                side_effect=_run_side_effect(secret_tool=spawn_error, scripted=_EXIT_ZERO),
            ) as run:
                self.assertIs(is_authenticated(), True)
                self.assertEqual(run.call_args.args[0][:2], [_VENV_PYTHON, "-c"])

    def test_probe_is_repeated_on_every_call(self):
        # Nothing is cached: a user who runs `agy` mid-session must be
        # authenticated on the very next call, so every call probes again.
        for stderr, expected in ((_SECRET_TOOL_STDERR_HIT, True), (b"", False)):
            with self.subTest(hit=bool(stderr)), _no_token_file(), _probe_env(), patch(
                "process.subprocess.run", return_value=_secret_tool_result(stderr)
            ) as run:
                self.assertIs(is_authenticated(), expected)
                self.assertIs(is_authenticated(), expected)
                self.assertEqual(run.call_count, 2)

    def _assert_secret_tool_call(self, kwargs):
        # stdout closed: it carries `secret = <credential>`. stderr captured as
        # bytes: the session locale is never decoded.
        self.assertIs(kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertIs(kwargs.get("stderr"), subprocess.PIPE)
        self.assertFalse(kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines"))
        self.assertLessEqual(kwargs.get("timeout"), 5)


class ScriptedProbeTests(unittest.TestCase):
    """The secretstorage fallback and its interpreter cascade."""

    def test_only_exit_zero_authenticates(self):
        for result, expected in ((_EXIT_ZERO, True), (_EXIT_THREE, False), (_EXIT_TWO, False)):
            with self.subTest(script_exit_code=result.returncode), _no_token_file(), _probe_env(
                secret_tool=None, executable=()
            ), patch("process.subprocess.run", return_value=result) as run:
                self.assertIs(is_authenticated(), expected)
                argv = run.call_args.args[0]
                self.assertEqual(argv[:2], [_VENV_PYTHON, "-c"])
                self.assertEqual(argv[2], process._KEYRING_PROBE_SCRIPT)
                self._assert_scripted_call(run.call_args.kwargs)

    def test_cascade_moves_to_the_next_interpreter_on_exit_two(self):
        # An interpreter without secretstorage (exit 2) is skipped, not fatal.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/bin/python3",)
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_ZERO])) as run:
            self.assertIs(is_authenticated(), True)
            self.assertEqual([call.args[0][0] for call in run.call_args_list], ["/usr/bin/python3", _VENV_PYTHON])

    def test_cascade_stops_on_definitive_miss(self):
        # Exit 3 answers the question: no further interpreter can do better.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/bin/python3", "/usr/local/bin/python3")
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_THREE, _EXIT_THREE, _EXIT_THREE])) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual([call.args[0][0] for call in run.call_args_list], ["/usr/bin/python3"])

    def test_cpython_exit_code_one_is_not_a_verdict(self):
        # CPython exits 1 on an uncaught exception, which happens BEFORE the
        # probe script's own try/except — a broken or ancient interpreter too
        # new-incompatible to run it. That is "cannot answer", so the cascade
        # must continue; reading it as a definitive miss would silently report
        # "not authenticated" while a working interpreter sits one entry later.
        with _no_token_file(), _probe_env(secret_tool=None, executable=("/usr/bin/python3",)), patch(
            "process.subprocess.run",
            side_effect=_run_side_effect(scripted=[_EXIT_BROKEN_INTERPRETER, _EXIT_ZERO]),
        ) as run:
            self.assertIs(is_authenticated(), True)
            self.assertEqual([c.args[0][0] for c in run.call_args_list], ["/usr/bin/python3", _VENV_PYTHON])

    def test_cascade_exhausted_is_false(self):
        # Every interpreter answers 2: no secretstorage anywhere in the cascade.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/bin/python3", "/usr/local/bin/python3")
        ), patch(
            "process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_TWO, _EXIT_TWO])
        ) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual(
                [call.args[0][0] for call in run.call_args_list],
                ["/usr/bin/python3", _VENV_PYTHON, "/usr/local/bin/python3"],
            )

    def test_cascade_survives_spawn_failure(self):
        # A TimeoutExpired is indistinguishable from "cannot probe": keep going.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/bin/python3",)
        ), patch(
            "process.subprocess.run",
            side_effect=_run_side_effect(scripted=[subprocess.TimeoutExpired([_PYTHON3], 5), _EXIT_ZERO]),
        ) as run:
            self.assertIs(is_authenticated(), True)
            self.assertEqual([call.args[0][0] for call in run.call_args_list], ["/usr/bin/python3", _VENV_PYTHON])

    def test_cascade_dedupes_interpreters_resolving_to_the_same_path(self):
        # which('python3') resolving to an entry already in the list must not
        # run the same interpreter twice.
        with _no_token_file(), _probe_env(
            secret_tool=None,
            bare_python="/usr/bin/python3",
            executable=("/usr/bin/python3", "/usr/local/bin/python3"),
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_TWO])) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual(
                [call.args[0][0] for call in run.call_args_list],
                ["/usr/bin/python3", "/usr/local/bin/python3"],
            )

    def test_cascade_dedupes_by_realpath_not_by_spelling(self):
        # Two absolute spellings of the same file must not spawn the same
        # interpreter twice. realpath is pinned by _probe_env (identity for
        # unknown paths), so host symlinks never decide the outcome.
        for bare_python in ("/venv/bin/python3", "/usr/bin/python3.3"):
            realpath_map = {"/venv/bin/python3": "/usr/bin/python3", bare_python: "/usr/bin/python3"}
            with self.subTest(bare_python=bare_python), _no_token_file(), _probe_env(
                secret_tool=None,
                bare_python=bare_python,
                executable=("/usr/bin/python3", "/usr/local/bin/python3"),
                realpath_map=realpath_map,
            ), patch(
                "process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_ZERO])
            ) as run:
                self.assertIs(is_authenticated(), True)
                self.assertEqual(
                    [call.args[0][0] for call in run.call_args_list],
                    ["/usr/bin/python3", "/usr/local/bin/python3"],
                )

    def test_cascade_skips_non_executable_trailing_entry(self):
        # An absolute entry without the executable bit is skipped, never spawned.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/bin/python3",)
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_TWO])) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual([call.args[0][0] for call in run.call_args_list], ["/usr/bin/python3", _VENV_PYTHON])

    def test_cascade_drops_relative_which_results(self):
        # A PATH entry of "" or "." makes which() return a RELATIVE path, and
        # spawning that verbatim would execute <cwd>/python3 — an arbitrary
        # local file. Such an entry is dropped, not spawned, and the next
        # absolute entry answers instead.
        for bare_python in ("python3", "./python3", "bin/python3"):
            with self.subTest(bare_python=bare_python), _no_token_file(), _probe_env(
                secret_tool=None,
                bare_python=bare_python,
                executable=("/usr/bin/python3", "/usr/local/bin/python3"),
            ), patch(
                "process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_ZERO])
            ) as run:
                self.assertIs(is_authenticated(), True)
                self.assertEqual(
                    [call.args[0][0] for call in run.call_args_list],
                    ["/usr/bin/python3", "/usr/local/bin/python3"],
                )

    def test_cascade_skips_non_executable_first_entry(self):
        # os.access(path, X_OK) is the gate, not mere existence: an absolute
        # entry that exists but is not executable is skipped, never spawned.
        with _no_token_file(), _probe_env(
            secret_tool=None, executable=("/usr/local/bin/python3",)
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_ZERO])) as run:
            self.assertIs(is_authenticated(), True)
            self.assertEqual(
                [call.args[0][0] for call in run.call_args_list], [_VENV_PYTHON, "/usr/local/bin/python3"]
            )

    def test_cascade_skips_unresolvable_bare_name(self):
        # which('python3') returning None means "not on PATH": skip the entry.
        with _no_token_file(), _probe_env(
            secret_tool=None,
            bare_python=None,
            executable=("/usr/bin/python3", "/usr/local/bin/python3"),
        ), patch("process.subprocess.run", side_effect=_run_side_effect(scripted=[_EXIT_TWO, _EXIT_TWO])) as run:
            self.assertFalse(is_authenticated())
            self.assertEqual(
                [call.args[0][0] for call in run.call_args_list],
                ["/usr/bin/python3", "/usr/local/bin/python3"],
            )

    def test_absolute_entries_are_gated_on_the_executable_bit(self):
        # Every absolute cascade entry is probed through os.access(path, X_OK)
        # with that exact mode constant: an existence-only check (F_OK) would
        # hand a non-executable interpreter to exec(), and a writability check
        # (W_OK) would accept one that is merely writable. Under the wraps
        # isolation pattern os.X_OK is a Mock, not the int, so the constant is
        # read while process.os is still the wrapped mock and compared by
        # identity — "the same constant twice" alone proves nothing about WHICH
        # bit is requested.
        with _no_token_file(), _probe_env(
            secret_tool=None, bare_python=None, executable=()
        ) as access_calls, patch("process.subprocess.run", return_value=_EXIT_ZERO) as run:
            expected_mode = process.os.X_OK
            self.assertFalse(is_authenticated())
            run.assert_not_called()
        self.assertEqual([path for path, _ in access_calls], ["/usr/bin/python3", "/usr/local/bin/python3"])
        self.assertEqual([mode for _, mode in access_calls], [expected_mode, expected_mode])

    def test_no_interpreter_at_all_is_false(self):
        # Nothing runnable anywhere: False, with no subprocess call attempted.
        with _no_token_file(), _probe_env(secret_tool=None, bare_python=None), patch(
            "process.subprocess.run", return_value=_EXIT_ZERO
        ) as run:
            self.assertFalse(is_authenticated())
            run.assert_not_called()

    def test_script_reads_only_metadata_and_never_the_default_collection(self):
        # Static zero-exfiltration guard on the frozen script's code (comments
        # are excluded, so a comment naming a forbidden call does not trip it):
        # only attribute/label metadata is read, and get_default_collection is
        # never called because it CREATES a "Default" collection — and can fire
        # a D-Bus unlock prompt — when the default alias does not exist.
        called = {
            node.attr for node in ast.walk(ast.parse(process._KEYRING_PROBE_SCRIPT)) if isinstance(node, ast.Attribute)
        }
        self.assertIn("search_items", called)
        self.assertIn("get_all_collections", called)
        self.assertIn("get_all_items", called)
        self.assertIn("get_label", called)
        for forbidden in ("get_secret", "retrieve_secret", "get_default_collection", "lookup"):
            self.assertNotIn(forbidden, called, forbidden)

    def test_script_uses_the_pinned_exit_codes_and_not_cpythons_own(self):
        # Codes 1 and 2 are CPython's own conventions (uncaught exception, CLI
        # misuse), which a broken or ancient interpreter raises before any of
        # the script's try/except runs; reusing either as the miss verdict would
        # make a broken interpreter look like a searched-but-empty keyring.
        script = process._KEYRING_PROBE_SCRIPT
        self.assertIn("sys.exit(0)", script)
        self.assertIn("sys.exit(3)", script)
        self.assertIn("sys.exit(2)", script)
        self.assertNotIn("sys.exit(1)", script)

    def _assert_scripted_call(self, kwargs):
        self.assertIs(kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertIs(kwargs.get("stderr"), subprocess.DEVNULL)
        self.assertLessEqual(kwargs.get("timeout"), 5)


class KeyringProbeScriptTests(unittest.TestCase):
    """The fallback script's exit-code contract, executed in-process.

    The script constant is exec'd with a stubbed ``secretstorage`` module
    injected into sys.modules, so neither D-Bus nor the real secretstorage
    package is needed. Items expose a get_secret() that fails loudly: the
    script exiting 0 proves it matched on metadata alone.
    """

    def test_exact_attribute_pair_hit_exits_zero(self):
        # The reporter's exact shape leads, and the pair is handed to
        # search_items() verbatim (server-side AND across all collections).
        item = _FakeItem(
            {"service": "gemini", "username": "antigravity", "xdg:schema": "org.freedesktop.Secret.Generic"},
            "Password for 'antigravity' on 'gemini'",
        )
        with self._stubbed(found_items=[item]) as calls:
            self.assertEqual(self._exec_script(), 0)
            self.assertEqual(calls["search"], [{"service": "gemini", "username": "antigravity"}])

    def test_empty_generator_from_search_is_not_a_match(self):
        # Regression: search_items() returns a GENERATOR, and a generator object
        # is always truthy, so an `if search_items(...):` guard reported "found"
        # on every host whose keyring merely answers. Proven live against
        # gnome-keyring + secretstorage 3.3.3 (an empty collection made the
        # probe authenticate); an empty generator must read as "not found".
        for found_items, expected in (([], 3), ([_agy_shaped_item()], 0)):
            with self.subTest(items=len(found_items)), self._stubbed(found_items=found_items):
                self.assertEqual(self._exec_script(), expected)

    def test_search_items_result_is_consumed_not_merely_truth_tested(self):
        # The mutation this pins: an unconsumed lazy generator looks exactly
        # like a hit if it is only tested for truthiness. The body below runs
        # only when something pulls from it, so exit 3 alone does not prove the
        # result was consumed — the started flag does.
        bookkeeping = {"started": False}

        def lazy_hits(bus, attributes):
            bookkeeping["started"] = True
            yield from ()

        stub = ModuleType("secretstorage")
        stub.dbus_init = lambda: "stub-session-bus"
        stub.search_items = lazy_hits
        stub.get_all_collections = lambda bus: iter(())
        with patch.dict(sys.modules, {"secretstorage": stub}):
            self.assertEqual(self._exec_script(), 3)
            self.assertTrue(bookkeeping["started"])

    def test_no_iterator_returning_call_sits_in_a_boolean_context(self):
        # Safety net, not the primary defense: assign-then-test (`h = f(); if h:`),
        # wrapper calls, and comprehensions slip past this static walk — the two
        # dynamic generator tests in the class below are the load-bearing pins.
        # Static companion to the empty-generator regression: a generator is
        # always truthy, so any secretstorage call that returns one must be
        # consumed, never used as a condition. (Verified against the installed
        # secretstorage 3.3.3 source: search_items(), get_all_collections() and
        # Collection.get_all_items() are all generators.)
        iterator_calls = {"search_items", "get_all_collections", "get_all_items"}
        tree = ast.parse(process._KEYRING_PROBE_SCRIPT)
        conditions = [node.test for node in ast.walk(tree) if isinstance(node, (ast.If, ast.IfExp))]
        conditions += [
            node.operand for node in ast.walk(tree) if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        ]
        for condition in conditions:
            for call in (node for node in ast.walk(condition) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Attribute):
                    self.assertNotIn(call.func.attr, iterator_calls, call.func.attr)

    def test_label_only_hit_exits_zero(self):
        # search_items() cannot see label-only hits, so the label branch walks
        # every collection instead — the deliberate extra permissiveness that
        # survives an agy service rename.
        item = _FakeItem({"service": "renamed", "username": "antigravity"}, "Password for 'antigravity' on 'renamed'")
        with self._stubbed(found_items=[], collections=[_FakeCollection([item])]) as calls:
            self.assertEqual(self._exec_script(), 0)
            self.assertEqual(calls["search"], [{"service": "gemini", "username": "antigravity"}])

    def test_no_items_anywhere_exits_three(self):
        with self._stubbed(found_items=[], collections=[]):
            self.assertEqual(self._exec_script(), 3)
        with self._stubbed(found_items=[], collections=[_FakeCollection([])]):
            self.assertEqual(self._exec_script(), 3)

    def test_unrelated_credentials_exit_three(self):
        unrelated = [
            [_FakeItem({"service": "gemini", "username": "bob"}, "Password for bob")],
            # gemini-cli's own credential: not an Antigravity session.
            [_FakeItem({"service": "gemini", "username": "claude"}, "Password for 'claude' on 'gemini'")],
            # Right username, wrong service, and no 'antigravity' in the label.
            [_FakeItem({"service": "other", "username": "antigravity"}, "some label")],
            [_FakeItem({"service": "chrome", "username": "alice"}, "Chrome Safe Storage")],
        ]
        for items in unrelated:
            with self.subTest(last=items[-1].label), self._stubbed(
                found_items=[], collections=[_FakeCollection(items)]
            ):
                self.assertEqual(self._exec_script(), 3)

    def test_unreadable_collection_does_not_hide_a_match_elsewhere(self):
        # A collection that cannot be read is skipped, so the label branch still
        # finds a match in another collection.
        hit = _FakeItem({"service": "gemini", "username": "antigravity"}, "Password for 'antigravity' on 'gemini'")
        with self._stubbed(found_items=[], collections=[_BrokenCollection(), _FakeCollection([hit])]):
            self.assertEqual(self._exec_script(), 0)

    def test_unusable_environment_exits_two(self):
        # Both the missing module and an outright D-Bus failure report "not
        # present": exit 2 is a verdict, not an error the caller has to unwind.
        with patch.dict(sys.modules, {"secretstorage": None}):
            self.assertEqual(self._exec_script(), 2)
        for search_error, collections_error in (
            (RuntimeError("no D-Bus session bus"), None),
            (None, RuntimeError("no D-Bus session bus")),
        ):
            with self.subTest(failed_in="search" if search_error else "collections"), self._stubbed(
                search_error=search_error, collections_error=collections_error
            ):
                self.assertEqual(self._exec_script(), 2)

    def test_secret_is_never_read(self):
        # The stub items raise if get_secret() is called, so exit 0 proves the
        # match came from attributes or the label alone.
        item = _FakeItem({"service": "gemini", "username": "antigravity"}, "Password for 'antigravity' on 'gemini'")
        with self._stubbed(found_items=[item]):
            self.assertEqual(self._exec_script(), 0)

    def _exec_script(self) -> int:
        """Run the frozen probe script in-process and return its exit code."""
        with self.assertRaises(SystemExit) as caught:
            exec(process._KEYRING_PROBE_SCRIPT, {"__name__": "keyring_probe_script"})
        return caught.exception.code

    @contextmanager
    def _stubbed(self, found_items=(), collections=(), search_error=None, collections_error=None):
        """A stubbed secretstorage module: search_items + get_all_collections."""
        stub = ModuleType("secretstorage")
        calls = {"search": []}

        def search_items(bus, attributes):
            calls["search"].append(attributes)
            if search_error is not None:
                raise search_error
            # A GENERATOR, exactly like the real secretstorage 3.x API: a
            # generator object is always truthy, so a stub that returned a list
            # would hide the truthiness bug this suite exists to catch.
            return (item for item in found_items)

        def get_all_collections(bus):
            if collections_error is not None:
                raise collections_error
            # Generator as well (secretstorage.collection.get_all_collections).
            return (collection for collection in collections)

        stub.dbus_init = lambda: "stub-session-bus"
        stub.search_items = search_items
        stub.get_all_collections = get_all_collections
        with patch.dict(sys.modules, {"secretstorage": stub}):
            yield calls


def _agy_shaped_item():
    """The reporter's live credential: service gemini, username antigravity."""
    return _FakeItem(
        {"service": "gemini", "username": "antigravity", "xdg:schema": "org.freedesktop.Secret.Generic"},
        "Password for 'antigravity' on 'gemini'",
    )


class _FakeItem:
    """Secret Service item exposing metadata only."""

    def __init__(self, attrs, label):
        self.attrs = attrs
        self.label = label

    def get_attributes(self):
        return dict(self.attrs)

    def get_label(self):
        return self.label

    def get_secret(self):
        raise AssertionError("the probe must never read the stored secret")


class _FakeCollection:
    def __init__(self, items):
        self._items = items

    def get_all_items(self):
        # A generator, matching the real API (Collection.get_all_items yields)
        # — the same standard the sibling stubs hold.
        yield from self._items


class _BrokenCollection:
    """A collection that cannot be read at all (e.g. no D-Bus session bus)."""

    def get_all_items(self):
        raise RuntimeError("no D-Bus session bus")


class GatingTests(unittest.TestCase):
    """When is_authenticated() must not probe the keyring at all."""

    def test_explicit_config_dir_blocks_the_probe(self):
        # ANTIGRAVITY_CONFIG_DIR is an override, not a hint: an empty explicit
        # dir stays unauthenticated even with a keyring credential present.
        with tempfile.TemporaryDirectory() as tmp, _no_token_file(), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(_SECRET_TOOL_STDERR_HIT)
        ) as run:
            with patch.dict(os.environ, {"ANTIGRAVITY_CONFIG_DIR": tmp}, clear=False):
                self.assertFalse(is_authenticated())
                run.assert_not_called()

    def test_other_platforms_do_not_probe(self):
        # A non-nt win32 host has no probe at all. darwin runs its own keychain
        # probe instead; test_macos_keychain.py pins that it never reaches this one.
        with _no_token_file(platform="win32"), _probe_env(), patch(
            "process.subprocess.run", return_value=_secret_tool_result(_SECRET_TOOL_STDERR_HIT)
        ) as run:
            self.assertFalse(is_authenticated())
            run.assert_not_called()

    def test_nt_still_routes_to_cmdkey(self):
        # PR #2's branch must keep serving Windows: sys.platform says "win32"
        # there, and the Windows probe (not the Linux one) must answer.
        listed = SimpleNamespace(stdout=b"Target: LegacyGeneric:target=gemini:antigravity\r\n", returncode=0)
        for result, expected in ((listed, True), (SimpleNamespace(stdout=b"* NONE *\r\n", returncode=0), False)):
            with self.subTest(expected=expected), _no_token_file(platform="win32", os_name="nt"), _probe_env(), patch(
                "process.subprocess.run", return_value=result
            ) as run:
                self.assertIs(is_authenticated(), expected)
                self.assertEqual(run.call_args.args[0], ["cmdkey", "/list:gemini:antigravity"])


class ConstantCoherenceTests(unittest.TestCase):
    """The argv, the stderr markers and the script cannot drift apart."""

    def test_argv_is_derived_from_the_service_and_username_constants(self):
        self.assertEqual(
            process._SECRET_TOOL_PROBE_ARGV,
            ["secret-tool", "search", "--all", "service", process._KEYRING_SERVICE, "username", process._KEYRING_USERNAME],
        )

    def test_stderr_markers_are_derived_from_the_same_constants(self):
        # Anchored at end-of-line, exactly as g_printerr emits them.
        expected = (
            b"attribute.service = " + process._KEYRING_SERVICE.encode() + b"\n",
            b"attribute.username = " + process._KEYRING_USERNAME.encode() + b"\n",
        )
        self.assertEqual(process._SECRET_TOOL_HIT_MARKERS, expected)
        # Bytes literals: the session locale is never decoded.
        self.assertTrue(all(isinstance(marker, bytes) for marker in process._SECRET_TOOL_HIT_MARKERS))

    def test_stderr_markers_are_anchored_so_a_prefix_cannot_match(self):
        # A longer attribute value that starts with ours must not satisfy the
        # marker, and the anchor is what does that work: the unanchored form
        # WOULD match such a line, the shipped one must not.
        for marker in process._SECRET_TOOL_HIT_MARKERS:
            padded = marker[:-1] + b"-cli\n"
            with self.subTest(marker=marker):
                self.assertNotIn(marker, padded)
                self.assertIn(marker[:-1], padded)

    def test_script_embeds_the_same_service_and_username(self):
        # The script text is a plain str constant with the constants injected,
        # so a rename of either half cannot leave one mechanism behind.
        script = process._KEYRING_PROBE_SCRIPT
        self.assertIn("service = %r" % process._KEYRING_SERVICE, script)
        self.assertIn("username = %r" % process._KEYRING_USERNAME, script)
        self.assertIn('"service": service, "username": username', script)
        self.assertIsInstance(script, str)


if __name__ == "__main__":
    unittest.main()
