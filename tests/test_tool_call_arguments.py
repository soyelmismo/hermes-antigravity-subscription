"""Hand-escaped <tool_call> arguments: repair, or withhold and ask again.

Corpus: the ten string-valued ``function.arguments`` that Hermes logged as
"Unrepairable tool_call arguments for terminal" on 2026-11-24
(gemini-4-neon via agy 1.5.15). Every frame decoded as JSON; only its
``arguments`` string did not: a backslash before a character JSON cannot
escape (``\\(`` ``\\|`` from find/grep), or a bare ``"`` inside a Python
f-string of a heredoc. Hermes treated each as a truncated reply and retried
with the same mistake (seven times in a row, then the turn failed).
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompt as prompt_module  # noqa: E402
from prompt import _lenient_json_loads, _normalize_tool_arguments, _parse_tool_block  # noqa: E402
import test_full_send_segments as segment_tests  # noqa: E402
from test_full_send_segments import _TurnProcess, _turn  # noqa: E402

CORPUS = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "tool_call_arguments_corpus.json").read_text(encoding="utf-8")
)["items"]

TERMINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "background": {"type": "boolean"},
        "timeout": {"type": "integer"},
        "workdir": {"type": "string"},
    },
    "required": ["command"],
}
# POSIX only. On a Windows runner ``bash`` resolves to the WSL launcher in
# System32 (no distribution installed: exit 1, empty stderr), not to Git's.
_BASH = os.name == "posix" and bool(shutil.which("bash"))

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": TERMINAL_SCHEMA}}]
SCHEMAS = {"terminal": TERMINAL_SCHEMA}


def _frame(arguments, call_id="call_1", name="terminal"):
    return "<tool_call>" + json.dumps(
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    ) + "</tool_call>"


class CorpusTests(unittest.TestCase):
    def test_corpus_is_invalid_json_as_logged(self):
        self.assertEqual(len(CORPUS), 10)
        for item in CORPUS:
            with self.subTest(item["id"]), self.assertRaises(json.JSONDecodeError):
                json.loads(item["arguments"])

    def test_every_corpus_item_repairs_to_the_intended_command(self):
        for item in CORPUS:
            with self.subTest(item["id"]):
                args, error = _normalize_tool_arguments("terminal", item["arguments"], SCHEMAS)
                self.assertIsNone(error)
                value = json.loads(args)
                self.assertEqual(list(value), ["command"])
                command = value["command"]
                if item["kind"] == "invalid-escape":
                    # The shell needs the literal backslash the model wrote.
                    self.assertTrue(re.search(r"\\[()|.]", command), command[:200])
                else:
                    body = re.search(r"<<'PY'\n(.*?)\nPY", command, re.S).group(1)
                    compile(body, item["id"], "exec")
                    self.assertIn('"DIR" if', body)
                if _BASH:
                    check = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
                    self.assertEqual(check.returncode, 0, check.stderr)

    def test_hermes_core_cannot_repair_the_corpus(self):
        try:
            from agent.message_sanitization import _repair_tool_call_arguments
        except Exception:
            self.skipTest("Hermes core not importable")
        for item in CORPUS:
            with self.subTest(item["id"]):
                self.assertEqual(_repair_tool_call_arguments(item["arguments"], "terminal"), "{}")


class LenientDecoderTests(unittest.TestCase):
    def test_valid_json_passes_through_byte_identical(self):
        raw = '{"command": "echo \\"hi\\"",   "timeout": 5}'
        self.assertEqual(_normalize_tool_arguments("terminal", raw, SCHEMAS), (raw, None))

    def test_invalid_escape_is_doubled(self):
        value, fixes = _lenient_json_loads('{"command": "find . \\( -name a \\)"}')
        self.assertEqual(value["command"], "find . \\( -name a \\)")
        self.assertEqual(fixes, {"escape": 2, "quote": 0})

    def test_escaped_single_quote_becomes_a_plain_quote(self):
        # JSON never escapes "'"; doubling the backslash would hand bash a
        # literal \' and sed a quoted script. bash -n cannot see that: it
        # only checks syntax, and \'s/a/b/\' is valid syntax.
        raw = r'{"command":"sed \'s/\./,/g\' f && echo done"}'
        args, error = _normalize_tool_arguments("terminal", raw, SCHEMAS)
        self.assertIsNone(error)
        self.assertEqual(json.loads(args)["command"], r"sed 's/\./,/g' f && echo done")

    def test_mixed_escapes_keep_shell_meaning(self):
        raw = r'{"command":"printf \'a.b|c\\n\' | sed \'s/\./,/g\' | grep -c \'a\|z\'"}'
        args, error = _normalize_tool_arguments("terminal", raw, SCHEMAS)
        self.assertIsNone(error)
        command = json.loads(args)["command"]
        self.assertEqual(command, r"printf 'a.b|c\n' | sed 's/\./,/g' | grep -c 'a\|z'")
        if _BASH:
            # Run it: a semantic check bash -n cannot make.
            out = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
            self.assertEqual(out.stdout.strip(), "1")

    def test_bare_quotes_inside_a_string_are_escaped(self):
        value, fixes = _lenient_json_loads('{"command": "python -c "print(1)""}')
        self.assertEqual(value["command"], 'python -c "print(1)"')
        self.assertEqual(fixes["quote"], 2)

    def test_truncated_arguments_are_not_guessed(self):
        args, error = _normalize_tool_arguments("terminal", '{"command": "ls -la', SCHEMAS)
        self.assertIn("not valid JSON", error)

    def test_quote_fix_that_merges_keys_is_rejected_by_the_schema(self):
        schema = {"edit": {"type": "object", "properties": {"path": {}, "content": {}}, "required": ["path", "content"]}}
        args, error = _normalize_tool_arguments("edit", '{"path":"x" "content"}', schema)
        self.assertIn("missing required content", error)

    def test_missing_comma_before_an_optional_key_is_not_repaired(self):
        """The schema check cannot see a merged key that was optional.

        Escaping the quote turned ``{"path": "/tmp/x" "content": "hi"}`` into a
        valid object whose only key, ``path``, held the rest of the text, and
        the call reached Hermes with that path.
        """
        schema = {"write_file": {
            "type": "object", "properties": {"path": {}, "content": {}}, "required": ["path"],
        }}
        for written in (
            '{"path": "/tmp/x" "content": "hi"}',
            '{"path": "/tmp/x" "content": "hi", "overwrite": true}',
            '{ "path" : "/tmp/x" , "content": "a" "mode": "append" }',
        ):
            with self.subTest(written=written):
                args, error = _normalize_tool_arguments("write_file", written, schema)
                self.assertIn("missing ',' between members", str(error))
                self.assertIn("not valid JSON", str(error))
                self.assertEqual(args, written)

        # No key after the quote: still a bare quote inside the string, and
        # every quote the model wrote is kept (the value ends unbalanced
        # because the model's own text does).
        value, fixes = _lenient_json_loads('{"command": "echo "done" && printf "%s" "x"}')
        self.assertEqual(value["command"], 'echo "done" && printf "%s" "x')
        self.assertEqual(fixes["quote"], 5)

    def test_non_object_arguments_are_rejected_after_repair(self):
        args, error = _normalize_tool_arguments("terminal", '"ls \\("', SCHEMAS)
        self.assertEqual(error, "arguments must be a JSON object")

    def test_frame_level_slip_with_object_arguments_is_repaired(self):
        body = '<tool_call>{"id": "c1", "type": "function", "function": {"name": "terminal", ' \
               '"arguments": {"command": "grep -n "a\\|b" f"}}}</tool_call>'
        calls, text = _parse_tool_block(body, tool_schemas=SCHEMAS)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].arguments_error)
        self.assertEqual(json.loads(calls[0].function.arguments)["command"], 'grep -n "a\\|b" f')

    def test_unrepairable_call_carries_arguments_error(self):
        calls, _text = _parse_tool_block(_frame('{"command": "ls'), tool_schemas=SCHEMAS)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].arguments_error)


class StreamCorrectionTests(segment_tests.SegmentedSendTests):
    """Protocol through the client, reusing the fake agy of the segment tests."""

    def setUp(self):
        super().setUp()
        os.environ.pop("HERMES_ANTIGRAVITY_SEGMENT_WEIGHT", None)

    MESSAGES = [{"role": "user", "content": "list the symlinks"}]

    def _tool_calls(self, chunks):
        calls = []
        for chunk in chunks:
            for choice in chunk.choices or []:
                calls.extend(choice.delta.tool_calls or [])
        return calls

    def test_repairable_corpus_frame_reaches_hermes_repaired(self):
        raw = CORPUS[4]["arguments"]
        proc = _TurnProcess([_turn(_frame(raw), {"input_tokens": 100, "output_tokens": 10})])
        self.pending.append(proc)
        c = self._client()
        chunks = list(c.chat.completions.create(model="gemini-4-neon", messages=self.MESSAGES, tools=TOOLS, stream=True))
        calls = self._tool_calls(chunks)
        self.assertEqual(len(calls), 1)
        json.loads(calls[0].function.arguments)
        self.assertEqual(len(proc.stdin.writes), 1)

    def test_unrepairable_call_is_withheld_and_the_model_asked_again(self):
        good = json.dumps({"command": "ls -la"})
        proc = _TurnProcess([
            _turn(_frame('{"command": "ls'), {"input_tokens": 100, "output_tokens": 10}),
            _turn(_frame(good, "call_9"), {"input_tokens": 260, "output_tokens": 30}),
        ])
        self.pending.append(proc)
        c = self._client()
        chunks = list(c.chat.completions.create(model="gemini-4-neon", messages=self.MESSAGES, tools=TOOLS, stream=True))
        calls = self._tool_calls(chunks)
        self.assertEqual([call.function.arguments for call in calls], [good])
        self.assertEqual(len(proc.stdin.writes), 2)
        correction = json.loads(proc.stdin.writes[1])["message"]["content"]
        self.assertIn("NOT executed", correction)
        self.assertIn("- terminal: arguments are not valid JSON", correction)
        usage = [ch.usage for ch in chunks if getattr(ch, "usage", None) is not None][-1]
        self.assertEqual(usage.prompt_tokens, 160)  # the re-emit call only
        self.assertEqual(c._worker_history, self.MESSAGES)
        self.assertNotIn(proc, self.terminated)

    def test_accepted_calls_are_not_repeated_after_correction(self):
        ok = json.dumps({"command": "pwd"})
        fixed = json.dumps({"command": "ls -la"})
        proc = _TurnProcess([
            _turn(_frame(ok, "call_1") + _frame('{"command": "ls', "call_2"), {"input_tokens": 100}),
            _turn(_frame(ok, "call_1") + _frame(fixed, "call_2"), {"input_tokens": 200}),
        ])
        self.pending.append(proc)
        c = self._client()
        chunks = list(c.chat.completions.create(model="gemini-4-neon", messages=self.MESSAGES, tools=TOOLS, stream=True))
        args = [call.function.arguments for call in self._tool_calls(chunks)]
        self.assertEqual(args, [ok, fixed])
        correction = json.loads(proc.stdin.writes[1])["message"]["content"]
        self.assertIn("The other 1 tool call(s) in that reply were accepted", correction)

    def test_two_failed_corrections_fail_the_turn_without_executing(self):
        bad = _turn(_frame('{"command": "ls'), {"input_tokens": 100})
        proc = _TurnProcess([bad, bad, bad])
        self.pending.append(proc)
        c = self._client()
        stream = c.chat.completions.create(model="gemini-4-neon", messages=self.MESSAGES, tools=TOOLS, stream=True)
        seen = []
        with self.assertRaisesRegex(RuntimeError, "not valid JSON and could not be repaired.*nothing was executed"):
            for chunk in stream:
                seen.append(chunk)
        self.assertEqual(self._tool_calls(seen), [])
        self.assertEqual(len(proc.stdin.writes), 3)
        self.assertFalse(proc.alive)

    def test_oneshot_cannot_ask_again_and_fails(self):
        proc = _TurnProcess([_turn(_frame('{"command": "ls'), {"input_tokens": 100})])
        self.pending.append(proc)
        c = self._client()
        self.assertTrue(c._worker_lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(RuntimeError, "could not be repaired"):
                list(c.chat.completions.create(model="gemini-4-neon", messages=self.MESSAGES, tools=TOOLS, stream=True))
        finally:
            c._worker_lock.release()


# The inherited segment tests already run in their own module.
for _name in [n for n in dir(segment_tests.SegmentedSendTests) if n.startswith("test_")]:
    setattr(StreamCorrectionTests, _name, None)


if __name__ == "__main__":
    unittest.main()
