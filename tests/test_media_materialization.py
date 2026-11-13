"""Media materialization hardening: SSRF, size caps, type gates, lifecycle.

Pins the security contract of `prompt._materialize_media`:

- local paths and file:// URIs are handed to agy ONLY when the extension is
  a known media extension (agy uploads what it reads to Google);
- remote fetches resolve every address and refuse loopback, private,
  link-local (cloud metadata), reserved, multicast and unspecified hosts,
  re-validating on every redirect hop;
- remote bodies are streamed under a hard size cap and must carry an
  image/*, video/* or audio/* content type;
- the materialized cache lives under the active Hermes home, not a
  hardcoded ~/.hermes, and the owning client deletes its files on close.

The local HTTP server tests bind 127.0.0.1 and monkeypatch only the host
gate, so the download path itself (content type, cap, memo) is exercised
offline and deterministically.
"""

from __future__ import annotations

import base64
import http.server
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import prompt as agy_prompt  # noqa: E402


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b""
    content_type = "image/png"
    hits = 0

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        self.send_response(200)
        self.send_header("Content-Type", type(self).content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):  # silence
        pass


class _LocalServer:
    """One-shot 127.0.0.1 server serving one fixed body/content-type."""

    def __init__(self, body: bytes, content_type: str):
        handler = type(
            "Handler",
            (_Handler,),
            {"body": body, "content_type": content_type, "hits": 0},
        )
        self.handler_class = handler
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/media"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def hits(self) -> int:
        return self.handler_class.hits


class LocalPathGateTests(unittest.TestCase):
    def setUp(self):
        agy_prompt._REMOTE_MEDIA_MEMO.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name: str, data: bytes = b"x") -> str:
        path = os.path.join(self.tmp.name, name)
        Path(path).write_bytes(data)
        return path

    def test_local_path_without_media_extension_refused(self):
        path = self._write("payload.exe")
        self.assertEqual(agy_prompt._materialize_media(path), ("", ""))

    def test_local_path_no_extension_refused(self):
        path = self._write("noextension")
        self.assertEqual(agy_prompt._materialize_media(path), ("", ""))

    def test_local_path_allowed_extensions(self):
        for name, kind in (
            ("a.png", "Image"),
            ("b.jpg", "Image"),
            ("c.mp4", "Video"),
            ("d.mp3", "Audio"),
        ):
            path = self._write(name)
            fpath, mtype = agy_prompt._materialize_media(path)
            self.assertEqual(mtype, kind, name)
            self.assertEqual(fpath, os.path.abspath(path))

    def test_vanished_allowed_path_still_emitted(self):
        fpath, mtype = agy_prompt._materialize_media("/nonexistent/dir/photo.png")
        self.assertEqual(mtype, "Image")
        self.assertEqual(fpath, "/nonexistent/dir/photo.png")

    def test_windows_drive_path_still_emitted(self):
        fpath, mtype = agy_prompt._materialize_media(r"C:\Users\sample.png")
        self.assertEqual(mtype, "Image")
        self.assertEqual(fpath, r"C:\Users\sample.png")

    def test_vanished_refused_path_dropped(self):
        self.assertEqual(
            agy_prompt._materialize_media("/nonexistent/dir/secrets.env"), ("", "")
        )


class FileURITests(unittest.TestCase):
    def test_file_uri_path(self):
        fpath, mtype = agy_prompt._materialize_media("file:///tmp/pic.png")
        self.assertEqual((fpath, mtype), ("/tmp/pic.png", "Image"))

    def test_file_uri_localhost(self):
        fpath, _ = agy_prompt._materialize_media("file://localhost/tmp/clip.mp4")
        self.assertEqual(fpath, "/tmp/clip.mp4")

    def test_file_uri_remote_host_refused(self):
        self.assertEqual(
            agy_prompt._materialize_media("file://evil.example.com/x.png"), ("", "")
        )

    def test_file_uri_bad_extension_refused(self):
        self.assertEqual(
            agy_prompt._materialize_media("file:///etc/passwd"), ("", "")
        )


class DataURLTests(unittest.TestCase):
    def setUp(self):
        agy_prompt._REMOTE_MEDIA_MEMO.clear()

    def test_media_data_url(self):
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
        fpath, mtype = agy_prompt._materialize_media(
            f"data:image/png;base64,{payload}"
        )
        self.assertTrue(fpath.endswith(".png"))
        self.assertEqual(mtype, "Image")

    def test_non_media_data_url_refused(self):
        payload = base64.b64encode(b"<html></html>").decode()
        self.assertEqual(
            agy_prompt._materialize_media(f"data:text/html;base64,{payload}"),
            ("", ""),
        )

    def test_oversized_data_url_refused(self):
        with patch.object(agy_prompt, "_MAX_DATA_URL_BYTES", 4):
            payload = base64.b64encode(b"123456789").decode()
            self.assertEqual(
                agy_prompt._materialize_media(f"data:image/png;base64,{payload}"),
                ("", ""),
            )


class SSRFGateTests(unittest.TestCase):
    def test_private_and_reserved_hosts_refused(self):
        for host in (
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "fc00::1",
            "ff02::1",
        ):
            self.assertFalse(
                agy_prompt._host_is_publicly_fetchable(host), host
            )

    def test_public_host_allowed(self):
        self.assertTrue(agy_prompt._host_is_publicly_fetchable("8.8.8.8"))

    def test_loopback_url_refused_before_network(self):
        self.assertEqual(
            agy_prompt._materialize_media("http://127.0.0.1/secret.png"), ("", "")
        )
        self.assertEqual(
            agy_prompt._materialize_media("http://169.254.169.254/latest/meta"),
            ("", ""),
        )

    def test_redirect_to_private_refused(self):
        handler = agy_prompt._SafeMediaRedirectHandler()
        req = urllib.request.Request("http://example.com/a.png")
        with self.assertRaises(urllib.error.URLError):
            handler.redirect_request(
                req, None, 302, "Found", {}, "http://127.0.0.1/b.png"  # pyright: ignore[reportArgumentType]
            )

    def test_redirect_to_public_allowed(self):
        handler = agy_prompt._SafeMediaRedirectHandler()
        req = urllib.request.Request("http://example.com/a.png")
        redirected = handler.redirect_request(
            req, None, 302, "Found", {}, "http://8.8.8.8/b.png"  # pyright: ignore[reportArgumentType]
        )
        self.assertIsNotNone(redirected)


class RemoteDownloadTests(unittest.TestCase):
    def setUp(self):
        agy_prompt._REMOTE_MEDIA_MEMO.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cache_patch = patch.object(
            agy_prompt, "_media_cache_dir", lambda: self.tmp.name
        )
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        gate_patch = patch.object(
            agy_prompt, "_host_is_publicly_fetchable", return_value=True
        )
        gate_patch.start()
        self.addCleanup(gate_patch.stop)

    def test_download_with_media_content_type(self):
        with _LocalServer(b"\x00" * 2048, "image/png") as server:
            fpath, mtype = agy_prompt._materialize_media(server.url)
            self.assertEqual(mtype, "Image")
            self.assertTrue(fpath.endswith(".png"))
            self.assertTrue(os.path.isfile(fpath))
            self.assertEqual(os.path.getsize(fpath), 2048)

    def test_download_with_non_media_content_type_refused(self):
        with _LocalServer(b"<html>hi</html>", "text/html") as server:
            self.assertEqual(agy_prompt._materialize_media(server.url), ("", ""))
        self.assertEqual(
            [name for name in os.listdir(self.tmp.name) if name.endswith(".part")],
            [],
        )

    def test_download_over_cap_refused_without_partial_files(self):
        body = b"\x00" * 4096
        with _LocalServer(body, "image/png") as server:
            with patch.object(agy_prompt, "_MAX_REMOTE_MEDIA_BYTES", 1024):
                self.assertEqual(agy_prompt._materialize_media(server.url), ("", ""))
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_download_memoized(self):
        with _LocalServer(b"\x89PNG" * 64, "image/png") as server:
            first = agy_prompt._materialize_media(server.url)
            second = agy_prompt._materialize_media(server.url)
            self.assertEqual(first, second)
            self.assertTrue(first[0])
            self.assertEqual(server.hits, 1)


class CacheDirAndLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name: str, data: bytes = b"x") -> str:
        path = os.path.join(self.tmp.name, name)
        Path(path).write_bytes(data)
        return path

    def test_cache_dir_derived_from_hermes_home(self):
        expected = os.path.join(
            str(agy_prompt.get_hermes_home()), "cache", "agy_media"
        )
        self.assertEqual(agy_prompt._media_cache_dir(), expected)

    def test_cache_dir_resolved_at_use_time_per_profile(self):
        # Review 5887913331 point 2: the cache dir must follow the ACTIVE
        # profile's home at every call, not a home captured at import.
        with patch.object(
            agy_prompt, "get_hermes_home", return_value=self.tmp.name
        ):
            self.assertEqual(
                agy_prompt._media_cache_dir(),
                os.path.join(self.tmp.name, "cache", "agy_media"),
            )

    def test_sink_registration(self):
        payload = base64.b64encode(b"\x89PNGfake").decode()
        sink: set[str] = set()
        fpath, _ = agy_prompt._materialize_media(
            f"data:image/png;base64,{payload}", sink
        )
        self.assertIn(fpath, sink)

    def test_passthrough_local_file_never_enters_sink(self):
        # Review 5887913331 point 1: a user-referenced local file goes to agy
        # BY PATH; registering it in the sink made session close delete the
        # user's own photo. Pass-through references must never enter the sink.
        sink: set[str] = set()
        fpath, mtype = agy_prompt._materialize_media(self._write("mine.png"), sink)
        self.assertEqual(mtype, "Image")
        self.assertNotIn(fpath, sink)
        self.assertTrue(os.path.isfile(fpath))

    def test_file_uri_never_enters_sink(self):
        sink: set[str] = set()
        fpath, _ = agy_prompt._materialize_media(
            "file://" + self._write("clip.mp4"), sink
        )
        self.assertNotIn(fpath, sink)

    def test_client_close_keeps_foreign_sink_paths(self):
        # Defense-in-depth layer: even a foreign path smuggled into the sink
        # is NOT deleted by close() unless it sits inside the plugin's own
        # media cache dir.
        import client as agy_client

        user_file = self._write("precious.png")
        cl = agy_client.AntigravityClient()
        try:
            cl._materialized_media.add(user_file)
        finally:
            cl.close()
        self.assertTrue(os.path.isfile(user_file))

    def test_client_close_deletes_materialized_media(self):
        import client as agy_client

        payload = base64.b64encode(b"\x89PNGfake").decode()
        sink: set[str] = set()
        fpath, _ = agy_prompt._materialize_media(
            f"data:image/png;base64,{payload}", sink
        )
        self.assertIn(fpath, sink)
        self.assertTrue(os.path.isfile(fpath))
        cl = agy_client.AntigravityClient()
        try:
            cl._materialized_media.add(fpath)
        finally:
            cl.close()
        self.assertFalse(os.path.exists(fpath))


class NativeMediaRootsTests(unittest.TestCase):
    def test_native_media_roots_resolved_at_use_time(self):
        # Review 5887913331 point 2: _path_is_native_media_target must
        # evaluate the roots of the ACTIVE profile home on every call.
        import types

        import stream as agy_stream

        module = types.ModuleType("hermes_constants")
        holder = {"home": "/home/one"}
        setattr(module, "get_hermes_home", lambda: holder["home"])
        saved = sys.modules.get("hermes_constants")
        sys.modules["hermes_constants"] = module
        try:
            inside = os.path.join(holder["home"], "cache", "x.png")
            self.assertTrue(
                agy_stream._path_is_native_media_target(inside, set())
            )
            holder["home"] = "/home/two"
            self.assertFalse(
                agy_stream._path_is_native_media_target(inside, set())
            )
            self.assertTrue(
                agy_stream._path_is_native_media_target(
                    os.path.join("/home/two", "uploads", "y.png"), set()
                )
            )
        finally:
            if saved is not None:
                sys.modules["hermes_constants"] = saved
            else:
                sys.modules.pop("hermes_constants", None)


if __name__ == "__main__":
    unittest.main()
