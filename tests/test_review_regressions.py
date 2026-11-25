"""Regression cases from the recovery and delivery review."""
import io
import json
import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import prompt
from delivery import DeliverySample, DeliveryWatch
from stream import AntigravityStream

NUDGE = ('You just executed tool calls but returned an empty response. '
         'Please process the tool results above and continue with the task.')
REQUEST = 'Inspect the current ledger, not the earlier deployment.'


class RecoveryObjectiveTests(unittest.TestCase):
    def history(self) -> list[dict[str, Any]]:
        return [{'role': 'user', 'content': REQUEST},
                {'role': 'tool', 'content': 'ledger rows'},
                {'role': 'assistant', 'content': '(empty)'},
                {'role': 'user', 'content': NUDGE}]

    def test_latest_user_skips_recovery_without_marker(self):
        self.assertEqual(prompt._latest_user_text(self.history()), REQUEST)

    def test_full_recovery_names_real_request_and_preserves_nudge(self):
        text = prompt._format_messages_as_prompt(self.history())
        head, transcript = text.split('Conversation transcript:', 1)
        self.assertIn(REQUEST, head)
        self.assertNotIn(NUDGE, head)
        self.assertIn(NUDGE, transcript)
        self.assertIn(REQUEST, text.split('### LATEST USER REQUEST TO ANSWER:')[-1])

    def test_delta_recovery_preserves_nudge_and_pending_request(self):
        text = prompt._format_delta_prompt(self.history()[-2:], pending_user_text=REQUEST)
        self.assertIn(NUDGE, text)
        self.assertIn(REQUEST, text.split('### LATEST USER REQUEST TO ANSWER:')[-1])

    def test_marker_is_respected_and_new_user_request_wins(self):
        history = self.history()
        history[-1] = {'role': 'user', 'content': 'synthetic recovery',
                       '_empty_recovery_synthetic': True}
        self.assertEqual(prompt._latest_user_text(history), REQUEST)
        history.append({'role': 'user', 'content': 'Stop and explain the last result'})
        text = prompt._format_messages_as_prompt(history)
        head = text.split('Conversation transcript:', 1)[0]
        self.assertIn('Stop and explain the last result', head)
        self.assertNotIn(REQUEST, head)

    def test_exact_match_only(self):
        text = 'Please explain this error: ' + NUDGE
        self.assertEqual(prompt._latest_user_text([{'role': 'user', 'content': text}]), text)

    def test_recovery_with_no_real_request_has_no_objective(self):
        text = prompt._format_messages_as_prompt([{'role': 'user', 'content': NUDGE}])
        self.assertNotIn('### ACTIVE OBJECTIVE:', text)
        self.assertNotIn('### LATEST USER REQUEST TO ANSWER:', text)
        self.assertIn(NUDGE, text)


class DeliveryRegressionTests(unittest.TestCase):
    def test_zero_usage_does_not_replace_valid_sample(self):
        watch = DeliveryWatch()
        watch.observe(64147, 95990)
        watch.observe(100000, 0)
        self.assertEqual(watch.last_sample, DeliverySample(64147, 95990))
        self.assertIsNotNone(watch.observe(126789, 96651))

    def test_different_route_does_not_compare(self):
        watch = DeliveryWatch()
        watch.observe(64147, 95990, sample_key='flash')
        self.assertIsNone(watch.observe(126789, 96651, sample_key='sonnet'))

    def stream(self, watch, input_tokens, cached_tokens, chars=126789):
        result = {'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'ok',
                  'usage': {'input_tokens': input_tokens, 'cache_read_tokens': cached_tokens}}}
        proc = MagicMock()
        proc.stdout = io.StringIO(json.dumps(result) + '\n')
        proc.poll.return_value = 0
        client = SimpleNamespace(_lock=threading.RLock(), _active_processes={proc},
                                 _terminate_process=lambda p: None)
        return AntigravityStream(proc=proc, client=client, model='flash', timeout=1,
                                 prompt_chars=chars, delivery_watch=watch)

    def test_delivery_and_chat_usage_both_include_cache_reads(self):
        watch = DeliveryWatch()
        watch.observe(64147, 95990)
        with self.assertNoLogs('delivery', level='WARNING'):
            chunks = list(self.stream(watch, 96651, 100000))
        sample = watch.last_sample
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.input_tokens, 196651)
        self.assertEqual(chunks[-1].usage.prompt_tokens, 196651)
        self.assertEqual(chunks[-1].usage.prompt_tokens_details.cached_tokens, 100000)

    def test_no_watch_means_delta_does_not_contaminate(self):
        watch = DeliveryWatch()
        watch.observe(64147, 95990)
        list(self.stream(None, 10000, 0, chars=15000))
        self.assertEqual(watch.last_sample, DeliverySample(64147, 95990))

    def test_stream_reports_comparable_full_send(self):
        watch = DeliveryWatch()
        watch.observe(64147, 95990)
        with self.assertLogs('delivery', level='WARNING'):
            list(self.stream(watch, 96651, 0))


class OutputLimitTests(unittest.TestCase):
    def test_output_limit_uses_provider_hook_in_real_core(self):
        import importlib.util
        import sys
        from pathlib import Path
        from agent.error_classifier import classify_api_error
        source = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "reviewed_provider", source / "__init__.py",
            submodule_search_locations=[str(source)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with patch('providers.get_provider_profile', return_value=module.antigravity_profile):
            for message in ('your previous response was cut off because it exceeded the output token limit',
                            'previous response was cut off'):
                verdict = classify_api_error(RuntimeError(message), provider='antigravity-subscription-directsdk')
                self.assertEqual(verdict.reason.value, 'server_error')
                self.assertTrue(verdict.retryable)
                self.assertFalse(verdict.should_compress)
            for message, expected in [('max_trajectory_tokens', 'context_overflow'),
                                      ('AI credits balance is too low', 'billing'),
                                      ('verify your account to continue', 'auth_permanent')]:
                self.assertEqual(classify_api_error(RuntimeError(message),
                    provider='antigravity-subscription-directsdk').reason.value, expected)


class ClientDeliveryWiringTests(unittest.TestCase):
    def test_full_delta_respawn_and_concurrent_oneshot_are_isolated(self):
        import tempfile
        from client import AntigravityClient
        from test_worker_usage import _mock_proc, _turn_events, _turn_lines
        first = [{'role': 'user', 'content': 'first'}]
        delta = first + [{'role': 'assistant', 'content': 'ok'},
                         {'role': 'tool', 'content': 'result'}]
        other = [{'role': 'user', 'content': 'new history'}]
        usage = {'input_tokens': 40000, 'output_tokens': 1}
        proc = _mock_proc(_turn_lines(
            _turn_events('worker-a', 'ok', usage),
            _turn_events('worker-a', 'ok', {'input_tokens': 85000, 'output_tokens': 2}),
        ))
        replacement = _mock_proc(_turn_lines(_turn_events('worker-b', 'ok', usage)))
        oneshot = _mock_proc(_turn_lines(_turn_events('one', 'ok', usage)), alive=False)
        with tempfile.TemporaryDirectory() as cwd, \
             patch('client.is_authenticated', return_value=True), \
             patch('process.resolve_real_token_path', return_value=None), \
             patch('process._link_macos_keychains'), \
             patch('client.start_stderr_drainer'), \
             patch('client._worker_idle_seconds', return_value=0), \
             patch('client.subprocess.Popen', side_effect=[proc, oneshot, replacement]):
            client = AntigravityClient(cwd=cwd, command='agy')
            with patch.object(client, '_resolve_model_and_effort', return_value=('flash', 'low')):
                try:
                    full = client.chat.completions.create(messages=first, stream=True)
                    self.assertIs(full.delivery_watch, client._delivery_watch)
                    list(full)
                    sample = client._delivery_watch.last_sample
                    incremental = client.chat.completions.create(messages=delta, stream=True)
                    self.assertIsNone(incremental.delivery_watch)
                    list(incremental)
                    self.assertEqual(client._delivery_watch.last_sample, sample)
                    client._worker_lock.acquire()
                    auxiliary = client.chat.completions.create(messages=other, stream=True)
                    self.assertIsNone(auxiliary.delivery_watch)
                    list(auxiliary)
                    client._worker_lock.release()
                    self.assertEqual(client._delivery_watch.last_sample, sample)
                    # A fresh full request uses fresh usage, not the dead
                    # worker's cumulative baseline. Comparable full samples survive.
                    new = client.chat.completions.create(messages=other, stream=True)
                    self.assertEqual(new.usage_baseline, {})
                    self.assertIs(new.delivery_watch, client._delivery_watch)
                    list(new)
                finally:
                    client.close()


if __name__ == '__main__':
    unittest.main()
