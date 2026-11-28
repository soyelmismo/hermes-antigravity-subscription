"""Loading the client must warm the openai type tree on the loading thread.

prompt.py imports agent.acp_openai_bridge lazily during the first turn, while Hermes
starts the title-generation thread in parallel. Two threads first-importing
openai.types.chat race and the loser sees a partially initialized module, so the
session title is lost. Importing the bridge at client load removes the race.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent


class TestClientWarmsOpenAITypes(unittest.TestCase):
    def test_client_import_loads_openai_chat_types(self):
        try:
            import openai.types.chat  # noqa: F401
        except Exception:
            self.skipTest("openai not installed in test runner environment")

        code = (
            "import sys\n"
            "assert 'openai.types.chat' not in sys.modules\n"
            "import client\n"
            "assert 'openai.types.chat' in sys.modules\n"
            "print('ok')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(PLUGIN_DIR), *sys.path])
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=PLUGIN_DIR, env=env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("ok", out.stdout)

    def test_client_import_succeeds_without_hermes_agent(self):
        code = (
            "import client\n"
            "print('ok')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PLUGIN_DIR)
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=PLUGIN_DIR, env=env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
