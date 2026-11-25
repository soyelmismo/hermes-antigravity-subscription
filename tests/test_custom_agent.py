"""Tests for the agent agy runs as (issue #16).

agy's default agent puts its own persona prompt and its run_command /
write_file / edit_file tools ahead of everything the plugin sends. The plugin
names an agent of its own instead, declared with excludeDefaultComponents, and
falls back to the default agent whenever that agent cannot be used: an agy
older than the verified floor, an opt-out, an unwritable home, or a spawn agy
rejects. The texts agy answers with are verbatim from agy 1.5.16.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import custom_agent
from client import AntigravityClient
from custom_agent import (
    AGENT_NAME,
    MIN_AGY_VERSION,
    agent_path,
    agy_version,
    custom_agent_args,
    note_custom_agent_rejection,
)

UNKNOWN_AGENT = f'unknown agent "{AGENT_NAME}" (available: )'
UNKNOWN_TOOL = (
    "failed to construct executor: failed to resolve components: "
    'unknown component: tool "read_file" not found in registry'
)


def _frontmatter(text: str) -> dict:
    """The definition's frontmatter as agy reads it: scalars, and ``tools`` as a list."""
    header = text.split("---\n")[1]
    fields: dict = {}
    key = None
    for line in header.splitlines():
        if line.startswith("  - ") and key:
            fields.setdefault(key, []).append(line[4:].strip())
            continue
        key, _, value = line.partition(":")
        if value.strip():
            fields[key] = value.strip().strip('"')
    return fields


class CustomAgentArgsTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        supported = patch("custom_agent.agy_version", return_value=MIN_AGY_VERSION)
        supported.start()
        self.addCleanup(supported.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("ANTIGRAVITY_DISABLE_CUSTOM_AGENT", None)

    def test_names_the_agent_and_writes_its_definition_under_the_isolated_home(self):
        self.assertEqual(custom_agent_args("agy", self.home), ["--agent", AGENT_NAME])
        definition = self.home / ".gemini" / "config" / "agents" / AGENT_NAME / "agent.md"
        self.assertEqual(agent_path(self.home), definition)
        fields = _frontmatter(definition.read_text(encoding="utf-8"))
        self.assertEqual(fields["name"], AGENT_NAME)
        self.assertTrue(fields["description"])
        self.assertEqual(fields["excludeDefaultComponents"], "true")
        self.assertEqual(fields["mainAgent"], "true")

    def test_the_only_agy_tool_left_is_the_media_read(self):
        custom_agent_args("agy", self.home)
        fields = _frontmatter(agent_path(self.home).read_text(encoding="utf-8"))
        self.assertEqual(fields["tools"], ["read_file"])

    def test_a_definition_changed_on_disk_is_restored_before_the_spawn(self):
        custom_agent_args("agy", self.home)
        agent_path(self.home).write_text("---\nname: other\n---\n", encoding="utf-8")
        self.assertEqual(custom_agent_args("agy", self.home), ["--agent", AGENT_NAME])
        self.assertIn(f"name: {AGENT_NAME}", agent_path(self.home).read_text(encoding="utf-8"))

    def test_an_agy_older_than_the_verified_floor_runs_its_default_agent(self):
        major, minor, patch_level = MIN_AGY_VERSION
        with patch("custom_agent.agy_version", return_value=(major, minor, patch_level - 1)):
            self.assertEqual(custom_agent_args("agy", self.home), [])
        self.assertFalse(agent_path(self.home).exists())

    def test_an_agy_whose_version_cannot_be_read_runs_its_default_agent(self):
        with patch("custom_agent.agy_version", return_value=None):
            self.assertEqual(custom_agent_args("agy", self.home), [])

    def test_the_opt_out_runs_the_default_agent(self):
        for value in ("1", "true", "YES", " on "):
            with patch.dict(os.environ, {"ANTIGRAVITY_DISABLE_CUSTOM_AGENT": value}):
                self.assertEqual(custom_agent_args("agy", self.home), [], value)
        with patch.dict(os.environ, {"ANTIGRAVITY_DISABLE_CUSTOM_AGENT": "0"}):
            self.assertEqual(custom_agent_args("agy", self.home), ["--agent", AGENT_NAME])

    def test_a_home_that_cannot_be_written_runs_the_default_agent(self):
        blocker = self.home / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        self.assertEqual(custom_agent_args("agy", blocker), [])


class RejectionTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        supported = patch("custom_agent.agy_version", return_value=MIN_AGY_VERSION)
        supported.start()
        self.addCleanup(supported.stop)

    def test_an_agent_agy_cannot_find_is_dropped_for_the_rest_of_the_process(self):
        self.assertEqual(custom_agent_args("agy", self.home), ["--agent", AGENT_NAME])
        self.assertTrue(note_custom_agent_rejection(UNKNOWN_AGENT))
        self.assertEqual(custom_agent_args("agy", self.home), [])

    def test_a_tool_name_agy_no_longer_has_drops_the_agent(self):
        self.assertTrue(note_custom_agent_rejection(UNKNOWN_TOOL))
        self.assertEqual(custom_agent_args("agy", self.home), [])

    def test_other_failures_keep_the_agent(self):
        for detail in ("RESOURCE_EXHAUSTED: quota exceeded", "subscriber fell behind updates", "", None):
            self.assertFalse(note_custom_agent_rejection(detail), detail)
        self.assertEqual(custom_agent_args("agy", self.home), ["--agent", AGENT_NAME])


class AgyVersionTests(unittest.TestCase):
    def setUp(self):
        # conftest replaces agy_version for every test; these exercise the real one.
        real = patch("custom_agent.agy_version", agy_version)
        real.start()
        self.addCleanup(real.stop)
        cache = patch.dict(custom_agent._versions, clear=True)
        cache.start()
        self.addCleanup(cache.stop)

    def test_reads_the_version_agy_prints_and_asks_once_per_command(self):
        printed = SimpleNamespace(stdout="1.5.16\n", stderr="", returncode=0)
        with patch("custom_agent.subprocess.run", return_value=printed) as run:
            self.assertEqual(agy_version("agy"), (1, 5, 16))
            self.assertEqual(agy_version("agy"), (1, 5, 16))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["agy", "--version"])

    def test_a_missing_or_silent_agy_has_no_version(self):
        with patch("custom_agent.subprocess.run", side_effect=FileNotFoundError("agy")):
            self.assertIsNone(agy_version("missing-agy"))
        with patch("custom_agent.subprocess.run", side_effect=subprocess.TimeoutExpired("agy", 5)):
            self.assertIsNone(agy_version("hung-agy"))
        with patch("custom_agent.subprocess.run", return_value=SimpleNamespace(stdout="", stderr="")):
            self.assertIsNone(agy_version("silent-agy"))


def _worker(events):
    """A fake agy worker that answers one turn with *events*."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    lines = [json.dumps(event) + "\n" for event in events]

    def readline():
        yield from lines
        while True:
            yield ""

    proc.stdout.readline.side_effect = readline()
    return proc


def _answering(text):
    return _worker([
        {"event": "init", "conversation_id": "conv-1"},
        {"event": "result", "result": {"status": "SUCCESS", "response": text, "usage": {}}},
    ])


class ClientSpawnTests(unittest.TestCase):
    def setUp(self):
        for target, value in (
            ("client.is_authenticated", True),
            ("client.resolve_real_token_path", None),
            ("client.resolve_agy_command", "agy"),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        keychains = patch("process._link_macos_keychains")
        keychains.start()
        self.addCleanup(keychains.stop)
        supported = patch("custom_agent.agy_version", return_value=MIN_AGY_VERSION)
        supported.start()
        self.addCleanup(supported.stop)
        import models

        models._catalog_cache = None
        self.addCleanup(lambda: setattr(models, "_catalog_cache", None))
        cwd = tempfile.TemporaryDirectory()
        self.addCleanup(cwd.cleanup)
        self.cwd = cwd.name
        self.spawns = []

    def _popen(self, *workers):
        remaining = iter(workers)

        def popen(cmd_args, **kwargs):
            if "models" in cmd_args:
                listing = MagicMock()
                listing.communicate.return_value = ("Gemini 3.8 Flash (High)\n", "")
                listing.returncode = 0
                return listing
            self.spawns.append(list(cmd_args))
            return next(remaining)

        return patch("subprocess.Popen", side_effect=popen)

    def _client(self, **kwargs):
        client = AntigravityClient(cwd=self.cwd, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_the_worker_is_spawned_as_the_custom_agent(self):
        client = self._client()
        with self._popen(_answering("hello")):
            list(client.chat.completions.create(
                model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True
            ))
        spawn = self.spawns[0]
        self.assertEqual(spawn[spawn.index("--agent") + 1], AGENT_NAME)
        self.assertTrue(agent_path(client._isolated_home).is_file())

    def test_a_worker_respawned_into_its_conversation_keeps_the_agent(self):
        client = self._client()
        first = [{"role": "user", "content": "turn 1"}]
        with self._popen(_answering("one"), _answering("two")):
            list(client.chat.completions.create(model="gemini-3.8-flash", messages=first, stream=True))
            client._terminate_worker(keep_conversation=True)
            second = first + [{"role": "tool", "tool_call_id": "call_1", "content": "result"}]
            list(client.chat.completions.create(model="gemini-3.8-flash", messages=second, stream=True))
        resumed = self.spawns[1]
        self.assertIn("--conversation", resumed)
        self.assertEqual(resumed[resumed.index("--agent") + 1], AGENT_NAME)

    def test_a_concurrent_oneshot_is_spawned_as_the_custom_agent_too(self):
        client = self._client()
        client._worker_lock.acquire()  # the worker is busy with another turn
        self.addCleanup(client._worker_lock.release)
        with self._popen(_answering("hello")):
            list(client.chat.completions.create(
                model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True
            ))
        spawn = self.spawns[0]
        self.assertNotIn("--input-format", spawn)
        self.assertEqual(spawn[spawn.index("--agent") + 1], AGENT_NAME)

    def test_an_agent_named_in_antigravity_args_is_left_alone(self):
        client = self._client(args=["--output-format", "stream-json", "--agent", "mine"])
        with self._popen(_answering("hello")):
            list(client.chat.completions.create(
                model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True
            ))
        self.assertEqual(self.spawns[0].count("--agent"), 1)
        self.assertNotIn(AGENT_NAME, self.spawns[0])

    def test_a_rejected_agent_fails_the_turn_and_the_retry_runs_the_default_agent(self):
        client = self._client()
        rejecting = _worker([
            {"event": "result", "result": {"status": "ERROR", "response": "", "error": UNKNOWN_AGENT}},
        ])
        messages = [{"role": "user", "content": "hi"}]
        with self._popen(rejecting, _answering("hello")):
            with self.assertRaisesRegex(RuntimeError, "unknown agent"):
                list(client.chat.completions.create(model="gemini-3.8-flash", messages=messages, stream=True))
            chunks = list(client.chat.completions.create(model="gemini-3.8-flash", messages=messages, stream=True))
        self.assertIn("--agent", self.spawns[0])
        self.assertNotIn("--agent", self.spawns[1])
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()
