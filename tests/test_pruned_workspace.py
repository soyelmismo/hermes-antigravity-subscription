"""The private directory can disappear under a long-lived client (issue #62).

Hermes points TMPDIR at a scratch dir and prunes entries with no writes for
24 h. ``AntigravityClient.__init__`` creates ``hermes_agy_*`` there once, an
idle compression client sees it deleted, and every later ``agy`` spawn fails
with ENOENT until the gateway restarts.
"""

from __future__ import annotations

import errno
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import AntigravityClient  # noqa: E402
import test_client_lifecycle as lifecycle  # noqa: E402

MODEL = "gemini-3.8-flash"
MESSAGES = [{"role": "user", "content": "first question"}]
FOLLOW_UP = MESSAGES + [
    {"role": "assistant", "content": "answer one"},
    {"role": "user", "content": "second question"},
]


def _proc(lines: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = lines + [""]
    return proc


class PrunedWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        lifecycle._pin_host_dependent_seams(self)

    def test_a_pruned_private_directory_is_recreated_before_the_spawn(self) -> None:
        """The real ``agy`` fails with ENOENT when its cwd is gone; so does this spawn."""
        spawned: list[str] = []
        turns = iter(
            [
                lifecycle._worker_turn_lines("conv-one", "answer one"),
                lifecycle._worker_turn_lines("conv-two", "answer two"),
            ]
        )

        def popen(*args, **kwargs):
            cwd = kwargs.get("cwd") or ""
            if not os.path.isdir(cwd):
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", cwd)
            spawned.append(cwd)
            return _proc(next(turns))

        client = AntigravityClient()
        self.addCleanup(client.close)
        # Model resolution and the agent lookup spawn `agy models` and `agy
        # --version` through the same Popen: warm them so the mock only sees workers.
        client._resolve_model_and_effort(MODEL, None)
        client._agent_args()
        with patch("subprocess.Popen", side_effect=popen):
            first = client.chat.completions.create(
                model=MODEL, messages=list(MESSAGES), stream=False
            )
            self.assertEqual(first.choices[0].message.content, "answer one")

            # What Hermes' scratch pruner does to an idle client.
            shutil.rmtree(client._cwd)
            self.assertFalse(os.path.exists(client._cwd))

            with self.assertLogs("client", level="WARNING") as logs:
                second = client.chat.completions.create(
                    model=MODEL, messages=list(FOLLOW_UP), stream=False
                )

        self.assertEqual(second.choices[0].message.content, "answer two")
        self.assertEqual(
            [r.getMessage() for r in logs.records if "was removed while the client was idle" in r.getMessage()],
            [
                f"Antigravity private directory {client._cwd} was removed while the client "
                "was idle; recreating it and dropping the agy conversation."
            ],
        )
        self.assertEqual(spawned, [client._cwd, client._cwd])
        self.assertTrue(os.path.isdir(client._cwd))
        # The conversation lived in the removed directory: nothing is resumed.
        self.assertEqual(client._worker_conversation_id, "conv-two")

    def test_an_existing_directory_is_left_alone(self) -> None:
        client = AntigravityClient(cwd=tempfile.mkdtemp(prefix="hermes_agy_test_"))
        self.addCleanup(shutil.rmtree, client._cwd, ignore_errors=True)
        isolated = client._isolated_home
        with patch("client.setup_isolated_home") as setup:
            client._restore_pruned_workspace()
        setup.assert_not_called()
        self.assertEqual(client._isolated_home, isolated)


if __name__ == "__main__":
    unittest.main()
