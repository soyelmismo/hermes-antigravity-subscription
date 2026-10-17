"""Unit tests for Antigravity subscription usage and quota tracking."""

import json
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add plugin root to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from usage import (
    SubscriptionUsage,
    UsageBucket,
    UsageGroup,
    _parse_iso_datetime,
    _query_agy_usage,
    _window_label,
    fetch_subscription_usage,
    format_countdown,
    format_reset_time,
    get_account_usage_snapshot,
    parse_agy_usage,
    render_usage_text,
    to_account_usage_snapshot,
)
import __init__ as plugin_module

SAMPLE_AGY_USAGE_JSON = {
    "conversation_id": "",
    "status": "SUCCESS",
    "response": "",
    "duration_seconds": 0,
    "num_turns": 0,
    "usage": {"input_tokens": 0, "output_tokens": 0},
    "command": {
        "name": "usage",
        "data": {
            "description": "User subscription quota limits",
            "groups": [
                {
                    "name": "Gemini Models",
                    "description": "Models within this group: Gemini Flash, Gemini Pro",
                    "buckets": [
                        {
                            "id": "gemini-weekly",
                            "name": "Weekly Limit Remaining",
                            "window": "weekly",
                            "remaining_fraction": 0.9849444627761841,
                            "reset_time": "2026-10-10T11:27:09Z",
                        },
                        {
                            "id": "gemini-5h",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 0.949386715888977,
                            "reset_time": "2026-10-03T16:27:09Z",
                        },
                    ],
                },
                {
                    "name": "Claude and GPT models",
                    "description": "Models within this group: Claude Opus, Claude Sonnet, GPT-OSS",
                    "buckets": [
                        {
                            "id": "3p-weekly",
                            "window": "weekly",
                            "remaining_fraction": 0.7443748116493225,
                            "reset_time": "2026-10-10T11:15:55Z",
                        },
                        {
                            "id": "3p-5h",
                            "window": "5h",
                            "remaining_fraction": 0.49769920110702515,
                            "reset_time": "2026-10-03T16:15:55Z",
                        },
                    ],
                },
            ],
        },
    },
}

# agy reports a third group on paid plans: Gemini 4 Argon keeps its own weekly pool, separate
# from the pooled Gemini Flash/Pro bucket. Both windows are "weekly", which is what made the
# two indistinguishable while every name containing "gemini" collapsed onto one prefix.
SAMPLE_AGY_USAGE_JSON_WITH_ARGON = json.loads(json.dumps(SAMPLE_AGY_USAGE_JSON))
SAMPLE_AGY_USAGE_JSON_WITH_ARGON["command"]["data"]["groups"].append(
    {
        "name": "Gemini 4 Argon",
        "description": (
            "For paid Pro plans, Argon quota is separate. Higher Argon quota (included as "
            "part of the larger Gemini quota bucket) is available on Ultra plans."
        ),
        "buckets": [
            {
                "id": "gemini-argon-wtu-weekly",
                "name": "Gemini 4 Argon Weekly Limit",
                "window": "weekly",
                "remaining_fraction": 1.0,
                "reset_time": "2026-10-24T14:06:32Z",
            },
        ],
    }
)

STREAM_JSON_SAMPLE = (
    '{"event":"init","conversation_id":""}\n'
    f'{json.dumps({"event": "command_result", "command": SAMPLE_AGY_USAGE_JSON["command"]})}\n'
    f'{json.dumps({"event": "result", "result": SAMPLE_AGY_USAGE_JSON})}\n'
)


class TestUsageParser(unittest.TestCase):
    def test_parse_valid_dict(self):
        usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        self.assertIsNotNone(usage)
        self.assertEqual(len(usage.groups), 2)

        gemini_group = usage.groups[0]
        self.assertEqual(gemini_group.name, "Gemini Models")
        self.assertEqual(len(gemini_group.buckets), 2)

        b_weekly = gemini_group.buckets[0]
        self.assertEqual(b_weekly.id, "gemini-weekly")
        self.assertEqual(b_weekly.window, "weekly")
        self.assertAlmostEqual(b_weekly.remaining_fraction, 0.98494446, places=5)
        self.assertEqual(b_weekly.reset_at, datetime(2026, 10, 10, 11, 27, 9, tzinfo=timezone.utc))

        claude_group = usage.groups[1]
        self.assertEqual(claude_group.name, "Claude and GPT models")
        self.assertEqual(len(claude_group.buckets), 2)

    def test_parse_valid_json_string(self):
        raw_str = json.dumps(SAMPLE_AGY_USAGE_JSON)
        usage = parse_agy_usage(raw_str)
        self.assertIsNotNone(usage)
        self.assertEqual(len(usage.groups), 2)

    def test_parse_stream_json(self):
        usage = parse_agy_usage(STREAM_JSON_SAMPLE)
        self.assertIsNotNone(usage)
        self.assertEqual(len(usage.groups), 2)
        self.assertEqual(usage.groups[0].name, "Gemini Models")

    def test_parse_zero_and_full_quota(self):
        payload = {
            "command": {
                "name": "usage",
                "data": {
                    "groups": [
                        {
                            "name": "Gemini Models",
                            "buckets": [
                                {
                                    "id": "gemini-zero",
                                    "window": "5h",
                                    "remaining_fraction": 0.0,
                                    "reset_time": "2026-10-03T12:00:00Z",
                                },
                                {
                                    "id": "gemini-full",
                                    "window": "weekly",
                                    "remaining_fraction": 1.0,
                                    "reset_time": "2026-10-10T12:00:00Z",
                                },
                            ],
                        }
                    ]
                },
            }
        }
        usage = parse_agy_usage(payload)
        self.assertIsNotNone(usage)
        buckets = usage.groups[0].buckets
        self.assertEqual(buckets[0].remaining_fraction, 0.0)
        self.assertEqual(buckets[1].remaining_fraction, 1.0)

    def test_parse_empty_and_corrupt_inputs(self):
        self.assertIsNone(parse_agy_usage(None))
        self.assertIsNone(parse_agy_usage(""))
        self.assertIsNone(parse_agy_usage("   "))
        self.assertIsNone(parse_agy_usage("invalid json string {"))
        self.assertIsNone(parse_agy_usage({}))
        self.assertIsNone(parse_agy_usage({"status": "ERROR"}))
        self.assertIsNone(parse_agy_usage({"command": {"name": "other"}}))
        self.assertIsNone(parse_agy_usage({"command": {"name": "usage", "data": {}}}))

    def test_parse_stream_json_alternative_envelopes(self):
        command = SAMPLE_AGY_USAGE_JSON["command"]

        # Envelope variant A: event/command_result wrapper carrying the command
        variant_a = '{"event":"command_result","command":' + json.dumps(command) + "}"
        usage_a = parse_agy_usage(variant_a)
        assert usage_a is not None
        self.assertEqual(len(usage_a.groups), 2)

        # Envelope variant B: result-wrapped full response object
        variant_b = json.dumps(
            {"event": "result", "result": SAMPLE_AGY_USAGE_JSON}
        )
        usage_b = parse_agy_usage(variant_b)
        assert usage_b is not None
        self.assertEqual(len(usage_b.groups), 2)

        # Envelope variant C: bare usage object with no command wrapper at all
        variant_c = json.dumps(command["data"])
        usage_c = parse_agy_usage(variant_c)
        assert usage_c is not None
        self.assertEqual(len(usage_c.groups), 2)

        # Mixed multi-line stream with unrelated events around the payload
        mixed = (
            '{"event":"init","conversation_id":"abc"}\n'
            '{"event":"progress","message":"querying"}\n'
            + variant_a
            + "\n"
            '{"event":"done"}\n'
        )
        usage_mixed = parse_agy_usage(mixed)
        assert usage_mixed is not None
        self.assertEqual(len(usage_mixed.groups), 2)

        # Lines that fail to parse or lack usage data are tolerated
        junk = "not json\n[1, 2, 3]\n" + variant_b
        usage_junk = parse_agy_usage(junk)
        assert usage_junk is not None
        self.assertEqual(len(usage_junk.groups), 2)


class TestDateTimeAndFormatting(unittest.TestCase):
    def test_parse_iso_datetime(self):
        dt = _parse_iso_datetime("2026-10-03T16:27:09Z")
        self.assertEqual(dt, datetime(2026, 10, 3, 16, 27, 9, tzinfo=timezone.utc))

        dt2 = _parse_iso_datetime("2026-10-03T16:27:09+00:00")
        self.assertEqual(dt2, datetime(2026, 10, 3, 16, 27, 9, tzinfo=timezone.utc))

        self.assertIsNone(_parse_iso_datetime(""))
        self.assertIsNone(_parse_iso_datetime(None))
        self.assertIsNone(_parse_iso_datetime("invalid-date"))

    def test_format_countdown(self):
        now = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)

        self.assertEqual(format_countdown(None, now), "unknown")

        # Passed reset time
        past = datetime(2026, 10, 3, 11, 59, 0, tzinfo=timezone.utc)
        self.assertEqual(format_countdown(past, now), "now")

        # Minutes
        in_25m = datetime(2026, 10, 3, 12, 25, 0, tzinfo=timezone.utc)
        self.assertEqual(format_countdown(in_25m, now), "in 25m")

        # Hours and minutes
        in_2h_15m = datetime(2026, 10, 3, 14, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(format_countdown(in_2h_15m, now), "in 2h 15m")

        # Days, hours and minutes (minutes included when days > 0)
        in_6d_16h_30m = datetime(2026, 10, 10, 4, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(format_countdown(in_6d_16h_30m, now), "in 6d 16h 30m")

        # Days without remaining hours/minutes
        in_3d = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(format_countdown(in_3d, now), "in 3d 0h 0m")

    def test_format_reset_time(self):
        self.assertEqual(format_reset_time(None), "unknown")
        dt = datetime(2026, 10, 3, 16, 27, 9, tzinfo=timezone.utc)
        self.assertEqual(format_reset_time(dt), "2026-10-03 16:27:09 UTC")

    def test_window_label(self):
        self.assertEqual(_window_label("Gemini Models", "5h"), "Gemini (5h)")
        self.assertEqual(_window_label("Gemini Models", "weekly"), "Gemini (Weekly)")
        self.assertEqual(_window_label("Claude and GPT models", "5h"), "Claude/GPT (5h)")
        self.assertEqual(_window_label("Claude and GPT models", "weekly"), "Claude/GPT (Weekly)")
        self.assertEqual(_window_label("Custom Group", "daily"), "Custom Group (daily)")

        # A group the payload adds later keeps its own name. agy reports "Gemini 4 Argon" as a
        # pool separate from "Gemini Models" (paid Pro plans), and both carry a weekly bucket:
        # collapsing every name that contains "gemini" onto one prefix made two distinct
        # quotas read as a single line in `hermes usage`.
        self.assertEqual(_window_label("Gemini 4 Argon", "weekly"), "Gemini 4 Argon (Weekly)")
        self.assertEqual(_window_label("Gemini 4 Argon", "5h"), "Gemini 4 Argon (5h)")
        self.assertNotEqual(
            _window_label("Gemini 4 Argon", "weekly"),
            _window_label("Gemini Models", "weekly"),
        )
        # Internal whitespace in the reported name does not change a known label.
        self.assertEqual(_window_label("  Gemini   Models  ", "weekly"), "Gemini (Weekly)")
        self.assertEqual(_window_label("", "weekly"), "Unknown (Weekly)")


class TestSnapshotAndRendering(unittest.TestCase):
    def test_to_account_usage_snapshot(self):
        usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        assert usage is not None
        snapshot = to_account_usage_snapshot(usage)

        self.assertEqual(snapshot.provider, "antigravity-subscription-directsdk")
        self.assertEqual(snapshot.source, "agy_cli")
        self.assertEqual(snapshot.title, "Antigravity subscription")
        self.assertEqual(len(snapshot.windows), 4)

        labels = [w.label for w in snapshot.windows]
        self.assertIn("Gemini (Weekly)", labels)
        self.assertIn("Gemini (5h)", labels)
        self.assertIn("Claude/GPT (Weekly)", labels)
        self.assertIn("Claude/GPT (5h)", labels)

        # Check used percent calculation: (1 - 0.9493867) * 100 ~ 5.06%
        w_gemini_5h = next(w for w in snapshot.windows if w.label == "Gemini (5h)")
        assert w_gemini_5h.used_percent is not None
        self.assertAlmostEqual(w_gemini_5h.used_percent, 5.0613, places=2)

    def test_snapshot_labels_stay_distinct_when_the_payload_adds_a_group(self):
        usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON_WITH_ARGON)
        assert usage is not None
        snapshot = to_account_usage_snapshot(usage)

        labels = [w.label for w in snapshot.windows]
        self.assertEqual(len(labels), 5)
        self.assertEqual(sorted(labels), sorted(set(labels)),
                         f"two quota windows must never share a label: {labels}")

        argon = next(w for w in snapshot.windows if w.label == "Gemini 4 Argon (Weekly)")
        assert argon.used_percent is not None
        self.assertAlmostEqual(argon.used_percent, 0.0, places=6)
        pooled = next(w for w in snapshot.windows if w.label == "Gemini (Weekly)")
        self.assertNotEqual(argon.reset_at, pooled.reset_at)
        self.assertEqual(argon.reset_at, datetime(2026, 10, 24, 14, 6, 32, tzinfo=timezone.utc))

    def test_render_usage_text(self):
        usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        assert usage is not None
        text = render_usage_text(usage)

        self.assertIn("Antigravity Subscription Quota:", text)
        self.assertIn("Gemini Models:", text)
        self.assertIn("Claude and GPT models:", text)
        self.assertIn("Weekly limit:", text)
        self.assertIn("5-hour limit:", text)
        self.assertIn("remaining", text)
        self.assertIn("used", text)
        self.assertIn("resets", text)


class TestCachingAndQuery(unittest.TestCase):
    def setUp(self):
        self._reset_cache()

    def tearDown(self):
        self._reset_cache()

    @staticmethod
    def _reset_cache():
        import usage
        with usage._cache_lock:
            usage._cached_usage = None
            usage._cached_timestamp = 0.0

    @patch("usage._query_agy_usage")
    def test_caching_and_force_refresh(self, mock_query):
        sample_usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        mock_query.return_value = sample_usage

        # First call hits mock_query
        u1 = fetch_subscription_usage()
        self.assertEqual(u1, sample_usage)
        self.assertEqual(mock_query.call_count, 1)

        # Second call returns cached without invoking mock_query
        u2 = fetch_subscription_usage()
        self.assertEqual(u2, sample_usage)
        self.assertEqual(mock_query.call_count, 1)

        # Force refresh invokes mock_query again
        u3 = fetch_subscription_usage(force_refresh=True)
        self.assertEqual(u3, sample_usage)
        self.assertEqual(mock_query.call_count, 2)

    @patch("usage._query_agy_usage")
    def test_stale_cache_fallback(self, mock_query):
        sample_usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        mock_query.return_value = sample_usage

        # Populate cache
        fetch_subscription_usage()
        self.assertEqual(mock_query.call_count, 1)

        # Simulate failure on refresh
        mock_query.return_value = None
        # force_refresh must NOT silently serve stale data — the caller asked
        # for fresh data, so failure should be surfaced as None.
        stale = fetch_subscription_usage(force_refresh=True)
        self.assertIsNone(stale)
        self.assertEqual(mock_query.call_count, 2)

        # Expire cache so the subsequent non-forced call bypasses the fresh fast path
        # and exercises the stale-cache fallback on query failure.
        import usage
        with usage._cache_lock:
            usage._cached_timestamp = time.time() - usage.USAGE_CACHE_TTL_SECONDS - 1.0

        # Non-forced callers still get the stale-cache fallback on failure.
        stale = fetch_subscription_usage()
        self.assertEqual(stale, sample_usage)
        self.assertEqual(mock_query.call_count, 3)

    @patch("usage._query_agy_usage")
    def test_single_flight_concurrent_queries_expired_cache(self, mock_query):
        sample_usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        query_started = threading.Event()
        release_query = threading.Event()

        def side_effect(*args, **kwargs):
            query_started.set()
            release_query.wait(timeout=5.0)
            return sample_usage

        mock_query.side_effect = side_effect

        n_threads = 5
        results = [None] * n_threads

        def worker(idx):
            results[idx] = fetch_subscription_usage()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]

        # Start first thread to trigger and hold query in flight
        threads[0].start()
        self.assertTrue(query_started.wait(timeout=2.0))

        # Start remaining threads while query is in flight
        for t in threads[1:]:
            t.start()

        # Brief sleep to allow waiting threads to attempt lock acquisition
        time.sleep(0.05)

        # Release the query
        release_query.set()

        for t in threads:
            t.join(timeout=3.0)

        # Only one subprocess query should be executed
        self.assertEqual(mock_query.call_count, 1)
        for res in results:
            self.assertEqual(res, sample_usage)

    @patch("usage._query_agy_usage")
    def test_single_flight_force_refresh_reuses_in_flight_query(self, mock_query):
        sample_usage = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        query_started = threading.Event()
        release_query = threading.Event()

        def side_effect(*args, **kwargs):
            query_started.set()
            release_query.wait(timeout=5.0)
            return sample_usage

        mock_query.side_effect = side_effect

        results = {}

        def background_worker():
            results["bg"] = fetch_subscription_usage()

        t_bg = threading.Thread(target=background_worker)
        t_bg.start()
        self.assertTrue(query_started.wait(timeout=2.0))

        # force_refresh waiting behind an in-flight refresh must reuse the new result
        def force_worker():
            results["force"] = fetch_subscription_usage(force_refresh=True)

        t_force = threading.Thread(target=force_worker)
        t_force.start()

        # Let force_worker enter and wait for the refresh lock
        time.sleep(0.05)

        # Release query
        release_query.set()

        t_bg.join(timeout=3.0)
        t_force.join(timeout=3.0)

        self.assertEqual(mock_query.call_count, 1)
        self.assertEqual(results["bg"], sample_usage)
        self.assertEqual(results["force"], sample_usage)

    @patch("usage.resolve_agy_command", return_value="/nonexistent/agy")
    @patch("usage.setup_isolated_home", return_value=("/tmp/home", False))
    @patch("usage.build_child_env", return_value={})
    def test_query_agy_file_not_found(self, mock_env, mock_home, mock_cmd):
        res = _query_agy_usage(timeout=1.0)
        self.assertIsNone(res)


class TestProviderIntegration(unittest.TestCase):
    def setUp(self):
        import usage
        with usage._cache_lock:
            usage._cached_usage = None
            usage._cached_timestamp = 0.0

    def tearDown(self):
        import usage
        with usage._cache_lock:
            usage._cached_usage = None
            usage._cached_timestamp = 0.0

    @patch("usage.fetch_subscription_usage")
    def test_provider_profile_fetch_account_usage(self, mock_fetch):
        mock_fetch.return_value = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        profile = plugin_module.antigravity_profile
        snapshot = profile.fetch_account_usage()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.provider, "antigravity-subscription-directsdk")
        self.assertEqual(len(snapshot.windows), 4)

    @patch("usage._query_agy_usage")
    def test_provider_profile_fetch_account_usage_returns_four_windows_from_fixture(self, mock_query):
        mock_query.return_value = parse_agy_usage(SAMPLE_AGY_USAGE_JSON)
        profile = plugin_module.antigravity_profile
        snapshot = profile.fetch_account_usage(force_refresh=True)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.provider, "antigravity-subscription-directsdk")
        self.assertEqual(len(snapshot.windows), 4)
        expected_labels = [
            "Gemini (Weekly)",
            "Gemini (5h)",
            "Claude/GPT (Weekly)",
            "Claude/GPT (5h)",
        ]
        self.assertEqual([w.label for w in snapshot.windows], expected_labels)
        for w in snapshot.windows:
            self.assertIsNotNone(w.used_percent)
            self.assertIsNotNone(w.reset_at)


if __name__ == "__main__":
    unittest.main()
