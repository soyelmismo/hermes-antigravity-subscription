import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from providers import get_provider_profile
from client import (
    AntigravityClient,
    _format_messages_as_prompt,
    _render_message_content,
    is_authenticated,
    resolve_agy_command,
)


class AntigravityPluginTests(unittest.TestCase):
    def test_provider_registration(self):
        profile = get_provider_profile("antigravity-subscription-directsdk")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "antigravity-subscription-directsdk")
        self.assertIn("antigravity", profile.aliases)
        self.assertIn("agy", profile.aliases)
        self.assertEqual(profile.auth_type, "external_process")
        self.assertEqual(profile.api_mode, "chat_completions")

    def test_alias_lookup(self):
        profile = get_provider_profile("antigravity")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "antigravity-subscription-directsdk")

    def test_command_resolution(self):
        cmd = resolve_agy_command()
        self.assertTrue(cmd.endswith("agy"))

    def test_auth_check(self):
        # On this environment, agy is logged in
        self.assertTrue(is_authenticated())

    def test_format_messages_prompt(self):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is the weather?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        prompt = _format_messages_as_prompt(messages, tools=tools)
        self.assertIn("Conversation transcript:", prompt)
        self.assertIn("System:\nYou are a helpful coding assistant.", prompt)
        self.assertIn("User:\nHello!", prompt)
        self.assertIn("Assistant:\nHi there!", prompt)
        self.assertIn("User:\nWhat is the weather?", prompt)
        self.assertIn("get_weather", prompt)

    def test_format_messages_with_tool_call_and_result(self):
        messages = [
            {"role": "user", "content": "Run calculator"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "name": "calculator", "content": "4"},
        ]
        prompt = _format_messages_as_prompt(messages)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("call_123", prompt)
        self.assertIn("Tool Result (call_123):\n4", prompt)

    def test_create_client_and_mock_turn(self):
        client = AntigravityClient(cwd="/tmp")
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "test-conv-1"}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": "Hello from mock"}}),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "Hello from mock\n",
                    "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cache_read_tokens": 0},
                },
            }),
        ]

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = io.StringIO("\n".join(fake_events) + "\n")
        mock_proc.stderr = io.StringIO("")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            res = client.chat.completions.create(
                model="gemini-3.8-flash-high",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            self.assertEqual(res.id, "test-conv-1")
            self.assertEqual(res.choices[0].message.content, "Hello from mock")
            self.assertEqual(res.choices[0].finish_reason, "stop")
            self.assertEqual(res.usage.prompt_tokens, 100)
            self.assertEqual(res.usage.completion_tokens, 10)
            self.assertEqual(res.usage.total_tokens, 110)

    def test_mock_tool_call_turn(self):
        client = AntigravityClient(cwd="/tmp")
        tool_call_obj = {"id": "call_99", "type": "function", "function": {"name": "test_tool", "arguments": json.dumps({"arg": 1})}}
        tool_call_body = f"<tool_call>{json.dumps(tool_call_obj)}</tool_call>"
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "test-tool-conv"}),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": tool_call_body,
                    "usage": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250},
                },
            }),
        ]

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = io.StringIO("\n".join(fake_events) + "\n")
        mock_proc.stderr = io.StringIO("")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            res = client.chat.completions.create(
                model="gemini-3.8-flash-high",
                messages=[{"role": "user", "content": "call test_tool"}],
                stream=False,
            )

            self.assertEqual(res.choices[0].finish_reason, "tool_calls")
            self.assertIsNone(res.choices[0].message.content)
            self.assertEqual(len(res.choices[0].message.tool_calls), 1)
            tc = res.choices[0].message.tool_calls[0]
            self.assertEqual(tc.id, "call_99")
            self.assertEqual(tc.function.name, "test_tool")
            self.assertEqual(json.loads(tc.function.arguments), {"arg": 1})

    def test_mock_stream_turn(self):
        client = AntigravityClient(cwd="/tmp")
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "stream-conv-1"}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": "Hello "}}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": "stream!"}}),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "Hello stream!\n",
                    "usage": {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
                },
            }),
        ]

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = io.StringIO("\n".join(fake_events) + "\n")
        mock_proc.stderr = io.StringIO("")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            stream = client.chat.completions.create(
                model="gemini-3.8-flash-high",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            chunks = list(stream)
            # Should have: delta("Hello "), delta("stream!"), finish("stop"), usage
            self.assertGreaterEqual(len(chunks), 4)

            # First data chunk
            self.assertEqual(chunks[0].choices[0].delta.content, "Hello ")
            self.assertIsNone(chunks[0].choices[0].finish_reason)

            # Second data chunk
            self.assertEqual(chunks[1].choices[0].delta.content, "stream!")
            self.assertIsNone(chunks[1].choices[0].finish_reason)

            # Finish chunk
            self.assertEqual(chunks[2].choices[0].finish_reason, "stop")

            # Usage chunk
            self.assertEqual(chunks[3].choices, [])
            self.assertEqual(chunks[3].usage.total_tokens, 55)

    def test_mock_stream_tool_call(self):
        client = AntigravityClient(cwd="/tmp")
        tool_call_obj = {"id": "call_stream_1", "type": "function", "function": {"name": "stream_tool", "arguments": json.dumps({"q": 42})}}
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "stream-tool-conv"}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": f"<tool_call>{json.dumps(tool_call_obj)}</tool_call>"}}),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": f"<tool_call>{json.dumps(tool_call_obj)}</tool_call>",
                    "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
                },
            }),
        ]

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = io.StringIO("\n".join(fake_events) + "\n")
        mock_proc.stderr = io.StringIO("")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        tools = [{"type": "function", "function": {"name": "stream_tool"}}]

        with patch("subprocess.Popen", return_value=mock_proc):
            stream = client.chat.completions.create(
                model="gemini-3.8-flash-high",
                messages=[{"role": "user", "content": "call stream_tool"}],
                tools=tools,
                stream=True,
            )

            chunks = list(stream)
            tool_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
            self.assertEqual(len(tool_chunks), 1)
            tc_delta = tool_chunks[0].choices[0].delta.tool_calls[0]
            self.assertEqual(tc_delta.id, "call_stream_1")
            self.assertEqual(tc_delta.function.name, "stream_tool")
            self.assertEqual(json.loads(tc_delta.function.arguments), {"q": 42})

            # Check finish reason
            finish_chunks = [c for c in chunks if c.choices and c.choices[0].finish_reason]
            self.assertEqual(finish_chunks[0].choices[0].finish_reason, "tool_calls")

    def test_base_model_and_effort_resolution(self):
        client = AntigravityClient(cwd="/tmp")
        # Base model + explicit effort
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash", "low"), ("gemini-3.8-flash-low", "low"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash", "medium"), ("gemini-3.8-flash-medium", "medium"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash", "high"), ("gemini-3.8-flash-high", "high"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash", "max"), ("gemini-3.8-flash-high", "high"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash", "none"), ("gemini-3.8-flash-low", "low"))

        # gemini-3.1-pro has only low and high
        self.assertEqual(client._resolve_model_and_effort("gemini-3.1-pro", "low"), ("gemini-3.1-pro-low", "low"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.1-pro", "medium"), ("gemini-3.1-pro-high", "high"))
        self.assertEqual(client._resolve_model_and_effort("gemini-3.1-pro", "high"), ("gemini-3.1-pro-high", "high"))

        # Suffix override: model had -high but effort was set to low
        self.assertEqual(client._resolve_model_and_effort("gemini-3.8-flash-high", "low"), ("gemini-3.8-flash-low", "low"))

        # Non-gemini model
        self.assertEqual(client._resolve_model_and_effort("claude-sonnet-4-6", "high"), ("claude-sonnet-4-6", "high"))

    def test_profile_supported_reasoning_efforts(self):
        profile = get_provider_profile("antigravity-subscription-directsdk")
        self.assertEqual(profile.supported_reasoning_efforts("gemini-3.8-flash"), ("low", "medium", "high"))
        self.assertEqual(profile.supported_reasoning_efforts("gemini-3.1-pro"), ("low", "high"))
        self.assertEqual(profile.supported_reasoning_efforts("gpt-oss-120b-medium"), ())

    def test_fetch_models_clean_base_names(self):
        profile = get_provider_profile("antigravity-subscription-directsdk")
        models = profile.fetch_models()
        self.assertIsNotNone(models)
        # Verify deduplicated clean base names
        self.assertIn("gemini-3.8-flash", models)
        self.assertIn("gemini-3.1-pro", models)
        self.assertNotIn("gemini-3.8-flash-high", models)
        self.assertNotIn("gemini-3.8-flash-medium", models)
    def test_security_default_args_omit_dangerous_permissions(self):
        client = AntigravityClient(cwd="/tmp")
        self.assertNotIn("--dangerously-skip-permissions", client._args)
        self.assertIn("--disable-slash-commands", client._args)
        self.assertIn("--output-format", client._args)

    def test_native_tool_step_neutralization_in_stream(self):
        client = AntigravityClient(cwd="/tmp")
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "test-sec-conv"}),
            json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "run_command"}}),
        ]
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [f"{line}\n" for line in fake_events] + [""]
        mock_proc.stdin = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), patch.object(client, "_terminate_process") as mock_term:
            stream = client.chat.completions.create(
                model="gemini-3.8-flash",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            chunks = list(stream)
            mock_term.assert_called_once_with(mock_proc)

    def test_temp_directory_isolation_and_cleanup(self):
        client = AntigravityClient()
        temp_dir = client._cwd
        self.assertTrue(Path(temp_dir).is_dir())
        self.assertIn("hermes_agy_", temp_dir)
        client.close()
        self.assertFalse(Path(temp_dir).exists())

    def test_messages_match_prefix_and_delta(self):
        from client import _messages_match_prefix, _format_delta_prompt

        history = [
            {"role": "system", "content": "You are Hermes"},
            {"role": "user", "content": "Run command"},
        ]
        incoming_same = list(history)
        self.assertFalse(_messages_match_prefix(history, incoming_same))

        incoming_extended = list(history) + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "sh", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        self.assertTrue(_messages_match_prefix(history, incoming_extended))

        delta = _format_delta_prompt(incoming_extended[2:])
        self.assertIn("<tool_call>", delta)
        self.assertIn("Tool Result (c1):\nok", delta)
        self.assertIn("Continue the conversation from the latest tool result.", delta)

    def test_session_worker_reuse_on_continuation(self):
        client = AntigravityClient(cwd="/tmp")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "conv-worker"}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "done", "usage": {}}}),
        ]
        mock_proc.stdout.readline.side_effect = [f"{line}\n" for line in fake_events] * 2 + [""]

        with patch("subprocess.Popen", return_value=mock_proc):
            # Turn 1
            msgs1 = [{"role": "user", "content": "first turn"}]
            stream1 = client.chat.completions.create(model="gemini-3.8-flash", messages=msgs1, stream=True)
            list(stream1)
            self.assertEqual(len(client._worker_history), 1)

            # Turn 2: continuation
            msgs2 = msgs1 + [{"role": "tool", "tool_call_id": "t1", "content": "res"}]
            stream2 = client.chat.completions.create(model="gemini-3.8-flash", messages=msgs2, stream=True)
            list(stream2)
            self.assertEqual(len(client._worker_history), 2)
            # Proc was not terminated between turns
            mock_proc.terminate.assert_not_called()

        client.close()
        mock_proc.terminate.assert_called()

    def test_concurrent_fallback_to_oneshot(self):
        client = AntigravityClient(cwd="/tmp")
        # Acquire worker lock manually to simulate an in-progress stream
        self.assertTrue(client._worker_lock.acquire(blocking=False))

        mock_oneshot_proc = MagicMock()
        mock_oneshot_proc.poll.return_value = 0
        mock_oneshot_proc.stdin = MagicMock()
        mock_oneshot_proc.stderr = io.StringIO("")
        fake_events = [
            json.dumps({"event": "init", "conversation_id": "oneshot-conv"}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "oneshot response"}}),
        ]
        mock_oneshot_proc.stdout = io.StringIO("\n".join(fake_events) + "\n")

        with patch("subprocess.Popen", return_value=mock_oneshot_proc):
            res = client.chat.completions.create(
                model="gemini-3.8-flash",
                messages=[{"role": "user", "content": "concurrent"}],
                stream=False,
            )
            self.assertEqual(res.choices[0].message.content, "oneshot response")

        client._worker_lock.release()
        client.close()

    def test_isolated_home_and_token_symlink(self):
        client = AntigravityClient()
        self.assertTrue(client._isolated_home.is_dir())
        self.assertTrue(client._isolated_gemini_dir.is_dir())
        symlinked_token = client._isolated_gemini_dir / "antigravity-oauth-token"
        self.assertTrue(symlinked_token.exists())
        temp_dir = client._cwd
        client.close()
        self.assertFalse(Path(temp_dir).exists())


if __name__ == "__main__":
    unittest.main()


