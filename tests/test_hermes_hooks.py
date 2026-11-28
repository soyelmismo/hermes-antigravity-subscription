"""Tests for Hermes provider hooks: setup_status, discover_models, resolve_aux_model, default_vision_model."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import models  # noqa: E402
import setup_hooks  # noqa: E402

AGY_MODELS = (
    "Fetching available models...\n"
    "gemini-4-neon-high\tGemini 4 Neon (High)\n"
    "gemini-4-neon-xhigh\tGemini 4 Neon (XHigh)\n"
    "gemini-4-neon-medium\tGemini 4 Neon (Medium)\n"
    "gemini-4-neon-low\tGemini 4 Neon (Low)\n"
    "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
    "claude-opus-5-5-high\tClaude Opus 5.5 (High)\n"
    "gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n"
)


class CatalogRowsTest(unittest.TestCase):
    def test_rows_group_efforts_and_keep_labels(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(
            by_id["gemini-4-neon"],
            {"id": "gemini-4-neon", "label": "Gemini 4 Neon", "note": "low, medium, high, xhigh"},
        )
        self.assertEqual(by_id["claude-opus-5-5"]["label"], "Claude Opus 5.5")
        self.assertEqual(by_id["claude-opus-5-5"]["note"], "high")
        self.assertNotIn("gemini-4-neon-xhigh", by_id)

    def test_empty_catalog(self):
        self.assertEqual(models.parse_catalog_rows(""), [])

    def test_label_keeps_non_effort_parenthetical_text(self):
        rows = models.parse_catalog_rows("gemini-4-neon-high\tGemini 4 Neon (Preview) (High)\n")
        self.assertEqual(rows[0]["label"], "Gemini 4 Neon (Preview)")


class SetupStatusTest(unittest.TestCase):
    def test_missing_binary_including_bare_fallback_string(self):
        # resolve_agy_command() returns literal "agy" when the binary is absent.
        with patch.object(setup_hooks, "_helpers", return_value=(lambda: "agy", lambda: False)), patch.object(
            setup_hooks.shutil, "which", return_value=None
        ), patch.object(setup_hooks, "_usage_snapshot") as mock_usage:
            status = setup_hooks.setup_status()
        self.assertFalse(status["available"])
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["detail"], setup_hooks.INSTALL_HINT)
        mock_usage.assert_not_called()

    def test_is_authenticated_is_enough(self):
        with patch.object(setup_hooks, "_helpers", return_value=(lambda: "/usr/bin/agy", lambda: True)), patch.object(
            setup_hooks, "_is_executable_command", return_value=True
        ), patch.object(setup_hooks, "_usage_snapshot", return_value=None):
            status = setup_hooks.setup_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["detail"], "")

    def test_usage_snapshot_adds_detail_and_covers_weekly_fallback(self):
        class Bucket:
            window = "5h"
            remaining_fraction = 0.62

        class Group:
            name = "Gemini Models"
            buckets = [Bucket()]

        class Snapshot:
            groups = [Group()]

        with patch.object(setup_hooks, "_helpers", return_value=(lambda: "/usr/bin/agy", lambda: False)), patch.object(
            setup_hooks, "_is_executable_command", return_value=True
        ), patch.object(setup_hooks, "_usage_snapshot", return_value=Snapshot()):
            status = setup_hooks.setup_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["detail"], "Gemini Models 5h: 62% left")

        class WeeklyBucket:
            window = "weekly"
            remaining_fraction = 0.80

        class WeeklyGroup:
            name = "Gemini Models"
            buckets = [WeeklyBucket()]

        class WeeklySnapshot:
            groups = [WeeklyGroup()]

        self.assertEqual(setup_hooks._quota_detail(WeeklySnapshot()), "Gemini Models weekly: 80% left")

    def test_logged_out_reports_login_hint(self):
        with patch.object(setup_hooks, "_helpers", return_value=(lambda: "/usr/bin/agy", lambda: False)), patch.object(
            setup_hooks, "_is_executable_command", return_value=True
        ), patch.object(setup_hooks, "_usage_snapshot", return_value=None):
            status = setup_hooks.setup_status()
        self.assertTrue(status["available"])
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["detail"], setup_hooks.LOGIN_HINT)

    def test_slow_auth_probe_is_bounded_and_usage_snapshot_still_answers(self):
        import time

        def slow_probe():
            time.sleep(2.0)
            return True

        with patch.object(setup_hooks, "_helpers", return_value=(lambda: "/usr/bin/agy", slow_probe)), patch.object(
            setup_hooks, "_is_executable_command", return_value=True
        ), patch.object(setup_hooks, "_AUTH_PROBE_DEADLINE", 0.2), patch.object(
            setup_hooks, "_usage_snapshot", return_value=None
        ):
            started = time.monotonic()
            status = setup_hooks.setup_status()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.5)
        self.assertFalse(status["logged_in"])

    def test_catalog_rows_cache_returns_copies_and_shares_probe_with_load_catalog(self):
        class Proc:
            returncode = 0
            stdout = AGY_MODELS

        models._catalog_cache = None
        models._rows_cache = None
        self.addCleanup(setattr, models, "_catalog_cache", None)
        self.addCleanup(setattr, models, "_rows_cache", None)
        with patch("models.subprocess.run", return_value=Proc()) as mock_run:
            rows = models.load_catalog_rows()
            rows[0]["label"] = "MUTATED"
            rows_again = models.load_catalog_rows()
            self.assertNotEqual(rows_again[0]["label"], "MUTATED")
            cat = models.load_catalog(only_if_stale=True)
            self.assertIn("gemini-4-neon", cat)
            self.assertEqual(mock_run.call_count, 1)

    def test_parse_mismatch_updates_neither_cache(self):
        # One stdout feeds both caches; a rows parse with an empty catalog parse
        # must not leave a fresh catalog next to stale picker rows.
        class Proc:
            returncode = 0
            stdout = AGY_MODELS

        models._catalog_cache = None
        models._rows_cache = None
        self.addCleanup(setattr, models, "_catalog_cache", None)
        self.addCleanup(setattr, models, "_rows_cache", None)
        with patch("models.subprocess.run", return_value=Proc()), patch.object(models, "parse_catalog", return_value={}):
            self.assertEqual(models.load_catalog_rows(), [])
            self.assertIsNone(models._catalog_cache)
            assert models._rows_cache is not None
            self.assertEqual(models._rows_cache[1], [])


class AuxAndVisionTest(unittest.TestCase):
    def test_aux_prefers_flash_low_then_neon_low(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.resolve_aux_model(), "gemini-3.8-flash-low")

        no_flash = [r for r in rows if r["id"] != "gemini-3.8-flash"]
        with patch.object(models, "load_catalog_rows", return_value=no_flash):
            self.assertEqual(setup_hooks.resolve_aux_model(), "gemini-4-neon-low")

    def test_default_vision_picks_gemini_4(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.default_vision_model(), "gemini-4-neon")

    def test_aux_uses_an_available_effort_instead_of_a_bare_base_id(self):
        # Models sold only as effort variants reject the bare id, so a base
        # without a "low" row must still resolve to one of its real variants.
        rows = [{"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash", "note": "medium, high"}]
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.resolve_aux_model(), "gemini-3.8-flash-medium")

    def test_aux_returns_empty_when_no_preferred_gemini_model_exists(self):
        rows = [{"id": "claude-opus-5-5", "label": "Claude Opus 5.5", "note": "high"}]
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.resolve_aux_model(), "")

    def test_aux_fallback_also_returns_an_effort_variant(self):
        # The generic gemini-4-* fallback must respect effort-only models too.
        rows = [{"id": "gemini-4-argon", "label": "Gemini 4 Argon", "note": "low, high"}]
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.resolve_aux_model(), "gemini-4-argon-low")

    def test_profile_hooks_forward_a_caller_timeout(self):
        from providers import get_provider_profile

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None
        package = type(profile).__module__
        prov_setup = importlib.import_module(f"{package}.setup_hooks")
        with patch.object(prov_setup, "resolve_aux_model", return_value="m1") as aux, patch.object(
            prov_setup, "default_vision_model", return_value="m2"
        ) as vision:
            self.assertEqual(profile.resolve_aux_model(timeout=3), "m1")
            self.assertEqual(profile.default_vision_model(timeout=2), "m2")
            self.assertEqual(profile.resolve_aux_model(vision=True, timeout=2), "m2")
        self.assertEqual(aux.call_args_list[0].kwargs, {"timeout": 3.0})
        self.assertEqual(vision.call_args_list[0].kwargs, {"timeout": 2.0})

    def test_profile_wires_hooks_and_handles_vision_flag(self):
        from providers import get_provider_profile

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None
        prov_models = sys.modules.get(f"{type(profile).__module__}.models", models)
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(prov_models, "load_catalog_rows", return_value=rows), patch.object(
            models, "load_catalog_rows", return_value=rows
        ):
            self.assertEqual(profile.resolve_aux_model(), "gemini-3.8-flash-low")
            self.assertEqual(profile.resolve_aux_model(vision=True), "gemini-4-neon")
            self.assertEqual(profile.default_vision_model(), "gemini-4-neon")
            self.assertEqual(profile.discover_models(), rows)

    def test_profile_setup_status_hides_exception_text(self):
        from providers import get_provider_profile

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None
        # The conftest loads the plugin under a package name, so the profile's
        # hooks resolve inside that package and need patching there.
        package = type(profile).__module__
        prov_setup = importlib.import_module(f"{package}.setup_hooks")
        prov_process = importlib.import_module(f"{package}.process")
        with patch.object(
            prov_setup,
            "setup_status",
            side_effect=RuntimeError("INTERNAL-PROBE-DETAIL"),
        ), patch.object(
            prov_setup, "_is_executable_command", return_value=True
        ), patch.object(
            prov_process, "resolve_agy_command", return_value="/usr/bin/agy"
        ):
            status = profile.setup_status()
        assert status is not None
        self.assertFalse(status["logged_in"])
        self.assertNotIn("INTERNAL-PROBE-DETAIL", status["detail"])
        self.assertIn("Setup check failed", status["detail"])


if __name__ == "__main__":
    unittest.main()
