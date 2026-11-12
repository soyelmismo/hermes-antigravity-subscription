"""Multimodal translation contract: synthetic notes -> canonical agy references.

Pins the bridge between Hermes' media shapes and agy's headless media contract
(``[Image](/abs/path)`` references in the turn text, pixels attached by agy's
own native read step). Runs against a temp media cache so no host state leaks
into the suite.

Verified failure chain this locks out (session transcripts, 2026-11-11):
a text-routed turn whose note survived as literal prose, or whose blanket
"native tools are disabled" preamble suppressed the read_file the media
pipeline requires, ends with the model answering "no puedo verlo" over media
it was shown.
"""

from __future__ import annotations

import base64
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import prompt as agy_prompt  # noqa: E402


def _png_data_url() -> str:
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


class _TempCacheTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_name = tmp.name
        patcher = patch.object(agy_prompt, "_MEDIA_CACHE_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)


class MediaNoteTranslationTests(_TempCacheTestCase):
    def test_gateway_text_mode_note_becomes_reference(self):
        text = (
            "[The user sent an image~ Here's what I can see:\n"
            "A black title card displays HERMES AGENT in white serif lettering.\n"
            "With brackets [like this] inside the description.]\n"
            "[If you need a closer look, use vision_analyze with "
            "image_url: /cache/img/desktop-upload.png ~]\n"
            "\n"
            "que ves aqui?"
        )
        out = agy_prompt._render_message_content(text)
        self.assertEqual(out.count("[Image](/cache/img/desktop-upload.png)"), 1)
        self.assertNotIn("Here's what I can see", out)
        self.assertNotIn("HERMES AGENT", out)  # description stripped, not fed
        self.assertNotIn("vision_analyze", out)
        self.assertIn("que ves aqui?", out)

    def test_analysis_failure_note_still_embeds_path(self):
        text = (
            "[The user sent an image but I couldn't quite see it "
            "this time (>_<) You can try looking at it yourself "
            "with vision_analyze using image_url: /cache/img/fail.png]\n"
            "hola"
        )
        out = agy_prompt._render_message_content(text)
        self.assertIn("[Image](/cache/img/fail.png)", out)
        self.assertNotIn("couldn't quite see", out)
        self.assertIn("hola", out)

    def test_cli_mixin_note_becomes_reference(self):
        text = (
            "[The user attached an image. Here's what it contains:\n"
            "A dashboard screenshot.]\n"
            "[If you need a closer look, use vision_analyze with "
            "image_url: /cache/img/cli.png]\n"
            "describe"
        )
        out = agy_prompt._render_message_content(text)
        self.assertIn("[Image](/cache/img/cli.png)", out)
        self.assertNotIn("dashboard screenshot", out)
        self.assertNotIn("closer look", out)

    def test_native_markers_collapse_to_one_reference(self):
        text = (
            "sin usar herramientas dime que ves\n\n"
            "[Image attached at: /cache/img/native.png]\n"
            "[screenshot]"
        )
        out = agy_prompt._render_message_content(text)
        self.assertEqual(out.count("[Image](/cache/img/native.png)"), 1)
        self.assertNotIn("Image attached at", out)
        self.assertNotIn("[screenshot]", out)
        self.assertIn("sin usar herramientas dime que ves", out)

    def test_video_note_becomes_video_reference(self):
        text = (
            "[The user sent a video attachment: 'clip.mp4'. "
            "It is saved at: /cache/vid/clip.mp4. "
            "Its content is not inlined here. If the user's request involves "
            "what the video contains, inspect or process it yourself — for "
            "example by passing the path to a video analysis or media tool — "
            "instead of asking the user to describe it. Only ask what to do "
            "with it if their intent is genuinely unclear.]\n\n"
            "resume el video"
        )
        out = agy_prompt._render_message_content(text)
        self.assertIn("[Video](/cache/vid/clip.mp4)", out)
        self.assertNotIn("saved at", out)
        self.assertNotIn("not inlined", out)
        self.assertIn("resume el video", out)

    def test_missing_file_still_referenced_by_path(self):
        text = (
            "[The user sent an image~ Here's what I can see:\ndesc]\n"
            "[If you need a closer look, use vision_analyze with "
            "image_url: /gone/missing.png]\n"
            "que es esto"
        )
        out = agy_prompt._render_message_content(text)
        self.assertIn("[Image](/gone/missing.png)", out)

    def test_prose_media_reference_untouched(self):
        text = "mira [Image](/real/photo.png) que acabo de subir"
        out = agy_prompt._render_message_content(text)
        self.assertIn("[Image](/real/photo.png)", out)

    def test_unterminated_note_head_keeps_text(self):
        text = "prefix [The user sent an image and then the text just continues"
        out = agy_prompt._render_message_content(text)
        self.assertIn("prefix", out)
        self.assertIn("and then the text just continues", out)

    def test_structured_image_part_materializes_data_url(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": _png_data_url()}},
        ]
        out = agy_prompt._render_message_content(content)
        self.assertRegex(out, r"\[Image\]\(" + re.escape(self.tmp_name))
        self.assertIn("describe this", out)


class MediaPromptAssemblyTests(_TempCacheTestCase):
    def test_media_turn_injects_viewing_protocol(self):
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "user",
                "content": (
                    "[The user sent an image~ Here's what I can see:\ndesc]\n"
                    "[If you need a closer look, use vision_analyze with "
                    "image_url: /cache/img/a.png]\nque ves?"
                ),
            },
        ]
        out = agy_prompt._format_messages_as_prompt(messages)
        self.assertIn("MEDIA VIEWING PROTOCOL", out)
        self.assertIn("[Image](/cache/img/a.png)", out)
        self.assertNotIn("Here's what I can see", out)
        self.assertLess(
            out.index("MEDIA VIEWING PROTOCOL"), out.index("[Image](/cache/img/a.png)")
        )
        self.assertIn("read_file", out)

    def test_non_media_turn_has_no_protocol(self):
        messages = [{"role": "user", "content": "hola que tal"}]
        out = agy_prompt._format_messages_as_prompt(messages)
        self.assertNotIn("MEDIA VIEWING PROTOCOL", out)

    def test_delta_prompt_tool_image_part_gets_protocol(self):
        messages = [
            {"role": "user", "content": "saca screenshot"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "vision_analyze", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Image loaded"},
                    {"type": "image_url", "image_url": {"url": _png_data_url()}},
                ],
            },
        ]
        out = agy_prompt._format_delta_prompt(
            messages, pending_user_text="saca screenshot"
        )
        self.assertIn("MEDIA VIEWING PROTOCOL", out)
        self.assertRegex(out, r"\[Image\]\([^\)]+\)")


class LazyModelCapabilitiesTests(unittest.TestCase):
    """The dynamic ``model_capabilities`` declaration behind native routing."""

    @classmethod
    def setUpClass(cls):
        import importlib.util as ilu

        spec = ilu.spec_from_file_location(
            "antigravity_plugin_entry_caps",
            plugin_dir / "__init__.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        assert spec and spec.loader  # noqa: S101 — test bootstrap
        module = ilu.module_from_spec(spec)
        sys.modules["antigravity_plugin_entry_caps"] = module
        spec.loader.exec_module(module)
        cls.caps = module._LazyModelCapabilities()

    def test_gemini_declares_image_input(self):
        entry = self.caps.get("gemini-3.8-flash")
        self.assertIsNotNone(entry)
        self.assertIn("image", entry["modalities"]["input"])

    def test_effort_suffix_resolves_to_same_declaration(self):
        self.assertEqual(self.caps.get("gemini-3.8-flash-high"), self.caps.get("gemini-3.8-flash"))

    def test_unknown_family_stays_undeclared(self):
        self.assertIsNone(self.caps.get("totally-unknown-model"))

    def test_declaration_uses_canonical_vision_key(self):
        # agent/models_dev._override_to_catalog_shape translates ``supports_vision``;
        # a modalities-only declaration is dropped by the catalog merge and the
        # capability silently resolves None -> the auxiliary describer wins.
        for model in ("gemini-3.8-flash", "gemini-4-hydrogen-medium", "claude-sonnet-4-6"):
            self.assertIs(self.caps.get(model).get("supports_vision"), True)


if __name__ == "__main__":
    unittest.main()
