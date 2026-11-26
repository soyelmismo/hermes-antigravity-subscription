"""Prompt construction, message translation, and tool-call parsing for Antigravity."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

try:
    from markdown_it import MarkdownIt
    from markdown_it.rules_inline.backticks import backtick as _orig_backtick
except ImportError:  # standalone import without hermes_agent / rich deps
    MarkdownIt = None  # type: ignore[assignment,misc]
    _orig_backtick = None  # type: ignore[assignment]

try:
    from hermes_constants import get_hermes_home
except ImportError:  # standalone use (tests, plugin dev)
    def get_hermes_home():
        return os.path.join(os.path.expanduser("~"), ".hermes")

logger = logging.getLogger(__name__)

# Media materialized for agy lives under the active Hermes home (profile-aware)
# and is deleted when the owning client session closes. The dir is resolved at
# USE time, not at import: one Hermes process can serve several profiles, and
# the cache must live under the home of the profile owning the session
# (review 5887913331, point 2).
def _media_cache_dir() -> str:
    return os.path.join(get_hermes_home(), "cache", "agy_media")


_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".ogv"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".opus", ".aac"}

# Remote fetch hardening: only public hosts, only media content types, and a
# hard size cap (the body is streamed, never read() whole).
_MAX_REMOTE_MEDIA_BYTES = 25 * 1024 * 1024
_MAX_DATA_URL_BYTES = 25 * 1024 * 1024

# mimetypes.guess_extension is OS-dependent (image/webp -> None on some
# systems); this explicit map keeps the extension gate deterministic.
_MEDIA_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/ogg": ".ogv",
}

# agy's headless media contract: pixels reach the model only through
# [Image](/abs/path) / [Video](/abs/path) markdown references in the turn text.
_MEDIA_REF_RE = re.compile(r"\[(?:Image|Video|Audio)\]\(([^)]+)\)")

_ATTACHMENT_ANNOTATION_RE = re.compile(
    # Bare [screenshot] (and any arg'd variant) carries no path: pure noise,
    # stripped as an annotation. [Image attached at: <path>] carries the ONLY
    # copy of the attachment path in a native-routed text message, so it is
    # NOT stripped here — _normalize_media_notes translates it into the
    # canonical reference. Stripping it first is what made native-routed
    # turns arrive at agy with no media reference at all.
    r"\n*\s*\[screenshot(?::\s*[^\]]+)?\]",
    re.IGNORECASE,
)


def _clean_rendered_text(text: str) -> str:
    if not text:
        return ""
    return _ATTACHMENT_ANNOTATION_RE.sub("", text).strip()


# --- Synthetic media notes -> canonical agy references -----------------------
#
# The Hermes core prepends several shapes of synthetic notes when a user turn
# carries media. In native routing the pixels travel as structured image_url
# parts; in text routing the turn text carries an auxiliary-vision description
# plus the local cache path. agy can only attach pixels from a markdown media
# reference, so every note shape is translated in place: the description block
# is dropped (the model receives the real pixels, a description of its own
# input is noise) and ONE canonical `[Image](/abs/path)` / `[Video](...)` is
# emitted where the note stood.

_NOTE_HEADS: tuple[tuple[str, str], ...] = (
    ("[The user sent a video attachment:", "Video"),
    ("[The user sent an audio file attachment:", "Audio"),
    ("[The user sent an image", "Image"),
    ("[The user attached an image", "Image"),
    ("[The tool result attached an image", "Image"),
    ("[The assistant attached an image", "Image"),
    ("[User sent an image:", "Image"),
    ("[Image attached at:", "Image"),
    # The paired "closer look" bracket that follows a description block (both
    # the gateway and the CLI variants). Consuming it here is what turns the
    # pair into exactly one reference instead of leaving the instruction
    # (and the path) as literal text the model can read.
    ("[If you need a closer look, use vision_analyze with", "Image"),
)

_VISION_NOTE_PATH_RE = re.compile(r"image_url:\s*([^\s\]~]+)")
_MEDIA_NOTE_SAVED_AT_RE = re.compile(r"It is saved at:\s*([^\s\]]+)")
_ATTACHED_AT_RE = re.compile(r"^\[Image attached at:\s*([^\]]+)\]\s*$")


def _consume_bracket_block(text: str, start: int) -> int:
    """Index just past the balanced ``[...]`` block opening at *start*, or -1.

    Balanced scanning instead of a lazy ``[^]]*`` regex: auxiliary vision
    descriptions may themselves contain ``]`` (markdown in the analysis), and
    the note only truly ends at depth zero.
    """
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


_MEDIA_LOCAL_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS


def _path_to_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "Video" if ext in _VIDEO_EXTS else "Image"


def _local_media_kind(path: str) -> Optional[str]:
    """Image/Video/Audio for a local path with an allowed media extension.

    Fail-closed gate: a local file without a known media extension is never
    handed to agy (agy uploads what it reads to Google).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "Image"
    if ext in _VIDEO_EXTS:
        return "Video"
    if ext in _AUDIO_EXTS:
        return "Audio"
    return None


def _media_reference_for(
    path: str, fallback_kind: str, media_sink: Optional[set] = None
) -> Optional[str]:
    """Canonical agy reference for a note-embedded path, or None.

    Per the normalization contract the reference is emitted by PATH, not by
    existence: a file that vanished (scratch pruning, replay from another
    host) must still surface as `[Image](/abs/path)` so the turn keeps its
    media intent instead of silently degrading to text-only. The extension
    gate applies regardless of existence: an unknown-extension path is
    dropped entirely.
    """
    if not path:
        return None
    kind = _local_media_kind(path)
    if kind is None:
        return None
    fpath, mtype = _materialize_media(path, media_sink)
    kind = mtype or kind
    return f"[{kind}]({fpath or path})"


def _normalize_media_notes(text: str, media_sink: Optional[set] = None) -> str:
    """Replace every synthetic media note with its canonical reference."""
    if not text or "[" not in text:
        return text
    out: list[str] = []
    emitted: set[str] = set()
    cursor = 0
    while True:
        hit: Optional[tuple[int, str, str]] = None
        for head, kind in _NOTE_HEADS:
            pos = text.find(head, cursor)
            if pos != -1 and (hit is None or pos < hit[0]):
                hit = (pos, head, kind)
        if hit is None:
            out.append(text[cursor:])
            break
        pos, head, kind = hit
        out.append(text[cursor:pos])
        end = _consume_bracket_block(text, pos)
        if end == -1:
            # Unterminated note head: keep the rest verbatim (an echo of the
            # marker in prose must not truncate the message).
            out.append(text[pos:])
            break
        block = text[pos:end]
        path: Optional[str] = None
        if head == "[Image attached at:":
            match = _ATTACHED_AT_RE.match(block.strip())
            path = match.group(1).strip() if match else None
        elif "image_url:" in block:
            match = _VISION_NOTE_PATH_RE.search(block)
            path = match.group(1).rstrip("~").strip() if match else None
        elif "saved at:" in block:
            match = _MEDIA_NOTE_SAVED_AT_RE.search(block)
            path = match.group(1).rstrip(".").strip() if match else None
        if path:
            reference = _media_reference_for(path, kind, media_sink)
            if reference and reference not in emitted:
                emitted.add(reference)
                out.append(reference)
        # Notes with no resolvable path are pure noise (failed analysis,
        # duplicated attachment marker): dropped, never re-described.
        cursor = end
    if not out:
        return text
    return "\n".join(part.strip("\n") for part in out if part and part.strip("\n")).strip() or ""


def _host_is_publicly_fetchable(hostname: str) -> bool:
    """True only when *hostname* resolves exclusively to public unicast addresses.

    SSRF gate for remote media fetches: loopback, private, link-local
    (which covers cloud metadata endpoints like 169.254.169.254), reserved,
    multicast and unspecified addresses make the host refuseable — every
    address the resolver returns counts, so a mixed-resolution hostname
    (one public, one private A record) is refused too.
    """
    if not hostname:
        return False
    try:
        infos = socket.getaddrinfo(hostname.rstrip("."), None)
    except (socket.gaierror, OSError, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip_text = str(info[4][0])
            address = ipaddress.ip_address(ip_text.split("%", 1)[0])
        except (ValueError, IndexError):
            return False
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


def _assert_fetchable_http_url(url: str) -> None:
    """Raise ValueError unless *url* is http(s) to a publicly resolvable host."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refused media URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname or ""
    if not _host_is_publicly_fetchable(hostname):
        raise ValueError(f"refused non-public media host: {hostname!r}")


class _SafeMediaRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop exactly like the initial URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _assert_fetchable_http_url(newurl)
        except ValueError as exc:
            raise urllib.error.URLError(f"media redirect refused: {exc}") from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_MEDIA_OPENER = urllib.request.build_opener(_SafeMediaRedirectHandler)

# Materialized-by-URL memo: prompt assembly renders the same messages more
# than once per turn (permission collection, prompt assembly, latest-user
# extraction), and without a memo every render would re-download. Entries are
# hints: a path whose file was deleted (session close) re-downloads on next
# use instead of handing agy a dead reference.
_REMOTE_MEDIA_MEMO: dict[str, tuple[str, str]] = {}
_REMOTE_MEDIA_MEMO_LIMIT = 512


def _is_plugin_owned_media_path(path: str) -> bool:
    """True only for files this plugin wrote into its own media cache dir.

    Pass-through references (the user's own photos, Hermes' gateway media
    cache files, anything handed over by bare path or file:// URI) resolve
    OUTSIDE the dir: they go to agy by reference and must never be registered
    for deletion, or session close would delete the user's own data (review
    5887913331, point 1). realpath comparison: the cache dir or the file may
    sit behind a symlink, and realpath works on vanished paths too.
    """
    if not path:
        return False
    cache_root = os.path.realpath(_media_cache_dir())
    real = os.path.realpath(path)
    return real == cache_root or real.startswith(cache_root + os.sep)


def _register_materialized(
    fpath: str, mtype: str, media_sink: Optional[set]
) -> tuple[str, str]:
    # Only files the plugin itself wrote into its media cache dir enter the
    # close-deletion sink; pass-through user files never do.
    if fpath and media_sink is not None and _is_plugin_owned_media_path(fpath):
        media_sink.add(fpath)
    return fpath, mtype


def _media_ext_for_mime(mime: str) -> Optional[str]:
    ext = mimetypes.guess_extension(mime)
    if ext == ".jpe":
        ext = ".jpg"
    return ext or _MEDIA_MIME_EXT.get(mime)


def _mime_kind(mime: str) -> str:
    if mime.startswith("video/"):
        return "Video"
    if mime.startswith("audio/"):
        return "Audio"
    return "Image"


def unquote_path(path: str) -> str:
    return urllib.parse.unquote(path or "")


def _stream_download_to_cache(resp: Any, ext: str) -> Optional[tuple[str, str]]:
    """Stream a media response into the cache dir under a hard size cap.

    Writes to a temp file and renames only after the whole body passed the
    cap, so an aborted download never leaves a partial file behind for agy
    to read. Returns (path, kind) or None when the cap is exceeded.
    """
    digest = hashlib.sha256()
    cache_dir = _media_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_REMOTE_MEDIA_BYTES:
                    logger.debug("Remote media exceeds %d byte cap", _MAX_REMOTE_MEDIA_BYTES)
                    return None
                digest.update(chunk)
                out.write(chunk)
        fpath = os.path.join(cache_dir, f"{digest.hexdigest()[:16]}{ext}")
        os.replace(tmp_path, fpath)
        return fpath, _mime_kind(resp.headers.get_content_type().lower())
    except Exception:
        logger.debug("Streaming media download failed", exc_info=True)
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _materialize_media(
    url_or_data: str, media_sink: Optional[set] = None
) -> tuple[str, str]:
    """Materialize a media reference (data URL, local path, or remote URL)
    into a local file so the official `agy` CLI can resolve it natively.

    Every input is gated fail-closed:
    - data URLs: only image/* and video/* mimes, capped decoded size;
    - local paths and file:// URIs: only when the extension is a known
      media extension (nothing unknown is ever handed to agy, which
      uploads what it reads);
    - http(s) URLs: only media content types, streamed with a hard size
      cap, only to publicly resolvable hosts, re-validated on redirects.

    Returns:
        (local_file_path, "Image" | "Video" | "Audio") or ("", "") when the
        reference is refused or cannot be materialized.
    """
    url = (url_or_data or "").strip()
    if not url:
        return "", ""

    # 1. Base64 data URL
    if url.startswith("data:"):
        try:
            header, b64_data = url.split(",", 1)
            mime = header.split(";")[0].replace("data:", "").strip().lower()
            if not (mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/")):
                logger.debug("Refused non-media data URL mime: %s", mime)
                return "", ""
            raw = base64.b64decode(b64_data)
            if len(raw) > _MAX_DATA_URL_BYTES:
                logger.debug("Refused oversized data URL media (%d bytes)", len(raw))
                return "", ""
            ext = _media_ext_for_mime(mime)
            if not ext:
                logger.debug("Refused data URL mime with no media extension: %s", mime)
                return "", ""
            h = hashlib.sha256(raw).hexdigest()[:16]
            cache_dir = _media_cache_dir()
            fpath = os.path.join(cache_dir, f"{h}{ext}")
            if not os.path.exists(fpath):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(fpath, "wb") as f:
                        f.write(raw)
                except OSError as exc:
                    logger.debug("Failed to write materialized data URL media: %s", exc)
                    return "", ""
            return _register_materialized(fpath, _mime_kind(mime), media_sink)
        except Exception as exc:
            logger.debug("Failed to materialize base64 media data URL: %s", exc)
            return "", ""

    # 2. file:// URI (local host only, no credentials/host indirection)
    if url.startswith("file://"):
        parsed = urllib.parse.urlsplit(url)
        if parsed.netloc not in ("", "localhost"):
            logger.debug("Refused file:// URI with host: %s", parsed.netloc)
            return "", ""
        local = unquote_path(parsed.path)
        kind = _local_media_kind(local)
        if kind is None:
            return "", ""
        # Emitted by path even when the file vanished: the media intent must
        # survive (agy's read of a missing file fails harmlessly).
        target_path = (
            local
            if local.startswith(("/", "\\")) or os.path.isabs(local)
            else os.path.abspath(local)
        )
        return _register_materialized(target_path, kind, media_sink)

    # 3. HTTP / HTTPS URL (SSRF-gated, capped, media content types only)
    if url.startswith(("http://", "https://")):
        try:
            _assert_fetchable_http_url(url)
        except ValueError as exc:
            logger.debug("Refused remote media URL %s: %s", url, exc)
            return "", ""
        memo = _REMOTE_MEDIA_MEMO.get(url)
        if memo is not None:
            if os.path.exists(memo[0]):
                return _register_materialized(memo[0], memo[1], media_sink)
            _REMOTE_MEDIA_MEMO.pop(url, None)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Hermes-Agent"}
            )
            with _MEDIA_OPENER.open(req, timeout=15) as resp:
                content_type = resp.headers.get_content_type().lower()
                if not (
                    content_type.startswith("image/")
                    or content_type.startswith("video/")
                    or content_type.startswith("audio/")
                ):
                    logger.debug(
                        "Refused remote media content type %s for %s",
                        content_type,
                        url,
                    )
                    return "", ""
                ext = _media_ext_for_mime(content_type)
                if not ext:
                    logger.debug(
                        "Refused remote media content type with no extension: %s",
                        content_type,
                    )
                    return "", ""
                result = _stream_download_to_cache(resp, ext)
            if result is None:
                return "", ""
            if len(_REMOTE_MEDIA_MEMO) >= _REMOTE_MEDIA_MEMO_LIMIT:
                for stale in list(_REMOTE_MEDIA_MEMO)[: _REMOTE_MEDIA_MEMO_LIMIT // 2]:
                    _REMOTE_MEDIA_MEMO.pop(stale, None)
            _REMOTE_MEDIA_MEMO[url] = result
            return _register_materialized(result[0], result[1], media_sink)
        except Exception as exc:
            logger.debug("Failed to download remote media URL %s: %s", url, exc)
            return "", ""

    # 4. Local filesystem path (checked last: URL shapes never reach it).
    # Extension gate only; emitted by path even when the file vanished.
    kind = _local_media_kind(url)
    if kind is not None:
        target_path = (
            url
            if (
                url.startswith(("/", "\\"))
                or os.path.isabs(url)
                or pathlib.PureWindowsPath(url).is_absolute()
            )
            else os.path.abspath(url)
        )
        return _register_materialized(target_path, kind, media_sink)

    return "", ""

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}

_PROMPT_PREAMBLE = (
    # agy embeds its own system prompt before the user's message. Some models
    # (claude-sonnet-4-6, gemini-3.6-flash) obey it and invoke native tools,
    # which headless mode soft-denies. A confrontational denial ("you are NOT
    # Antigravity") makes the model classify the preamble as prompt injection
    # and resist harder. This framing avoids contradicting agy's identity and
    # instead redirects the tool-calling protocol. (Issue #16)
    "You are operating in HEADLESS INFERENCE MODE. In this mode, all native function-calling / tool-use channels (run_command, write_to_file, view_file, edit_file, etc.) are disabled and will fail or be silently rejected. Never emit native function calls—even for Hermes tool names such as terminal, read_file, write_file, or patch.",
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent on the host side.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in plain text output—NEVER use the native functionCall / tool_use channel.",
    "- Each <tool_call> block must contain a single valid JSON object with 'id' (unique per call), 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}." ,
    "- 'arguments' must be a valid, properly JSON-escaped string encoding the function's arguments object.",
    "- Do NOT execute local shell commands or file operations directly; only output <tool_call> tags so Hermes can execute them safely.",
    "- NEVER simulate or hallucinate Tool Results (e.g. 'Tool Result (...):'). Hermes Agent executes tools externally and will supply real results in subsequent turns.",
    "- After emitting tool calls, STOP generating immediately. Do NOT generate results, execution output, or commentary after tool calls.",
    "- If no tool is needed, respond naturally with standard text.",
    "- When the user's latest message introduces a new question or task, do NOT continue prior completed tasks or tool loops. Address the user's latest request directly.",
)

# agy attaches media pixels through its own native read step: the model must
# call read_file/view_file on the referenced path, and the CLI then inlines the
# file as a media block. The blanket "native tools are disabled" preamble would
# otherwise forbid the one tool call the media pipeline requires (verified in
# session transcripts: reference present + read denied -> no pixels -> "no
# puedo verlo"). This section is injected ONLY on turns whose assembled prompt
# actually carries a media reference.
_MEDIA_VIEWING_SECTION = (
    "### MEDIA VIEWING PROTOCOL (applies ONLY to this turn's media references):\n"
    "Messages may contain markdown media references: [Image](/abs/path), [Video](/abs/path) or [Audio](/abs/path).\n"
    "These are real attachments, not prose. To see the media, call your native read_file (or view_file) "
    "tool on that exact path once — the runtime then attaches the file's pixels to your context, and you "
    "answer from what you see.\n"
    "This read is the ONLY permitted use of a native tool in this mode; every other native tool stays "
    "disabled. Never describe media you have not loaded this way, and never narrate loading steps — "
    "just read the file, then answer."
)


def _apply_media_viewing_protocol(sections: list[str]) -> list[str]:
    """Insert the media viewing section when the assembled prompt carries a reference."""
    if not any(_MEDIA_REF_RE.search(section) for section in sections):
        return sections
    insert_at = 0
    for index, section in enumerate(sections):
        if section.startswith("Conversation transcript:"):
            insert_at = index
            break
    sections.insert(insert_at, _MEDIA_VIEWING_SECTION)
    return sections


_TOOL_CALL_PREFIXES = tuple(
    "<tool_call>"[:i] for i in range(len("<tool_call>"), 0, -1)
)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _build_md_parser() -> Any:
    """CommonMark parser that records exact source spans on ``code_inline`` tokens."""
    if MarkdownIt is None or _orig_backtick is None:
        return None
    md = MarkdownIt("commonmark")

    def _tracked_backtick(state: Any, silent: bool) -> bool:
        start = state.pos
        n_before = len(state.tokens)
        ok = _orig_backtick(state, silent)
        if ok and not silent and len(state.tokens) > n_before:
            tok = state.tokens[-1]
            if tok.type == "code_inline":
                tok.meta["span"] = (start, state.pos)
        return ok

    md.inline.ruler.at("backticks", _tracked_backtick)
    return md


_MD_PARSER = _build_md_parser()


def _map_inline_spans(region: str, content: str, children: Sequence[Any]) -> list[tuple[int, int]]:
    """Map ``code_inline`` spans from *content* coordinates to *region* coordinates."""
    spans = [c.meta["span"] for c in children if c.type == "code_inline" and "span" in c.meta]
    if not spans:
        return []
    if region.startswith(content):
        return spans
    c_lines = content.split("\n")
    r_lines = region.splitlines(keepends=True)
    idx_map: list[int] = []
    r_pos = 0
    for idx, c_ln in enumerate(c_lines):
        r_ln = r_lines[idx] if idx < len(r_lines) else ""
        r_stripped = r_ln.rstrip("\r\n")
        lead = r_stripped.find(c_ln)
        if lead == -1:
            lead = 0
        for j in range(len(c_ln)):
            idx_map.append(r_pos + lead + j)
        if idx < len(c_lines) - 1:
            idx_map.append(r_pos + len(r_stripped))
        r_pos += len(r_ln)
    idx_map.append(r_pos)
    max_idx = len(idx_map) - 1
    return [(idx_map[min(s, max_idx)], idx_map[min(e, max_idx)]) for s, e in spans]


def _code_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted ``(start, end)`` char spans for all CommonMark code constructs."""
    if not text or _MD_PARSER is None:
        return []
    lines = text.splitlines(keepends=True)
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))
    spans: list[tuple[int, int]] = []
    for tok in _MD_PARSER.parse(text):
        if not tok.map:
            continue
        s_line, e_line = tok.map
        region_start = line_starts[s_line]
        region_end = line_starts[min(e_line, len(lines))]
        if tok.type in ("fence", "code_block"):
            spans.append((region_start, region_end))
        elif tok.type == "inline" and tok.children:
            region = text[region_start:region_end]
            for s, e in _map_inline_spans(region, tok.content, tok.children):
                spans.append((region_start + s, region_start + e))
    return spans


def _trailing_unclosed_fence(tokens: Sequence[Any], lines: list[str]) -> Any | None:
    """Return the trailing ``fence`` token if it remains unclosed at EOF.

    Walks past trailing container ``*_close`` tokens (e.g. ``blockquote_close``)
    and relies on ``markdown-it-py``'s own fence-closing decision: a closed
    fence excludes its closing line from ``tok.content`` (yielding at most
    ``e_line - s_line - 2`` body lines), whereas an unclosed fence includes all
    lines through EOF (``e_line - s_line - 1`` body lines).
    """
    for tok in reversed(tokens):
        if tok.type.endswith("_close"):
            continue
        if tok.type != "fence" or not tok.map:
            return None
        s_line, e_line = tok.map
        if e_line < len(lines):
            return None
        if e_line <= s_line + 1:
            return tok
        body_lines = len(tok.content.splitlines())
        return tok if body_lines >= e_line - s_line - 1 else None
    return None


class _MarkdownCodeState:
    """Track active CommonMark code context across stream chunks via ``markdown-it-py``.

    Retains only the active unclosed block (an open fence or the current
    paragraph since the last blank line) so incremental streaming stays bounded
    while preserving full CommonMark semantics.
    """

    def __init__(self, tail: str = "") -> None:
        self._tail = tail

    def clone(self) -> "_MarkdownCodeState":
        return _MarkdownCodeState(self._tail)

    def in_code(self) -> bool:
        if not self._tail or _MD_PARSER is None:
            return False
        lines = self._tail.splitlines(keepends=True)
        tokens = _MD_PARSER.parse(self._tail)
        return _trailing_unclosed_fence(tokens, lines) is not None

    def advance(self, text: str) -> None:
        """Advance markdown code state through *text* and compact completed blocks."""
        if not text or _MD_PARSER is None:
            return
        self._tail += text
        lines = self._tail.splitlines(keepends=True)
        tokens = _MD_PARSER.parse(self._tail)
        fence_tok = _trailing_unclosed_fence(tokens, lines)
        if fence_tok is not None:
            s_line = fence_tok.map[0]
            opener = lines[s_line]
            if not opener.endswith(("\n", "\r")):
                opener += "\n"
            trailing = (
                lines[-1]
                if len(lines) > s_line + 1 and not self._tail.endswith(("\n", "\r"))
                else ""
            )
            self._tail = opener + trailing
            return
        for sep in ("\r\n\r\n", "\n\n"):
            pos = self._tail.rfind(sep)
            if pos != -1:
                self._tail = self._tail[pos + len(sep):]
                return


def _find_unquoted_tool_call_open(
    text: str, state: _MarkdownCodeState | None = None
) -> int:
    """Return index of first ``<tool_call`` in *text* outside CommonMark code, or -1."""
    if "<tool_call" not in text:
        return -1
    prefix = state._tail if state is not None else ""
    offset = len(prefix)
    spans = _code_spans(prefix + text)
    span_idx = 0
    cursor = 0
    while True:
        idx = text.find("<tool_call", cursor)
        if idx == -1:
            return -1
        abs_pos = offset + idx
        while span_idx < len(spans) and spans[span_idx][1] <= abs_pos:
            span_idx += 1
        if span_idx >= len(spans) or abs_pos < spans[span_idx][0]:
            return idx
        cursor = idx + len("<tool_call")


def _find_tool_call_close(text: str, start: int = 0) -> int:
    """Index of the first ``</tool_call>`` outside a JSON string, or -1.

    A tool argument may legitimately contain the literal closing tag (for
    example a patch or a file whose content includes protocol markup). A plain
    ``str.find`` truncates the call at that literal and hands invalid JSON to
    the parser. This scanner tracks JSON double-quoted strings and backslash
    escapes, so only a tag at string depth zero closes the frame. It is the
    single frame boundary used by both the incremental stream splitter and
    :func:`_parse_tool_block`.
    """
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if text.startswith(_TOOL_CALL_CLOSE, index):
            return index
    return -1


def _render_message_content(content: Any, media_sink: Optional[set] = None) -> str:
    """Normalize multimodal or structured message content into a string.

    Synthetic media notes are translated to canonical agy references so the
    CLI's media pipeline can attach the real pixels (see _normalize_media_notes).
    Refused media (unknown extension, non-public host, non-media type) is
    dropped entirely: nothing unverifiable is ever handed to agy.
    """
    rendered: str
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            rendered = _clean_rendered_text(str(content.get("text") or ""))
        else:
            rendered = _clean_rendered_text(str(content.get("content") or ""))
        return _normalize_media_notes(rendered, media_sink)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                cleaned = _clean_rendered_text(item)
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    cleaned = _clean_rendered_text(str(item.get("text") or ""))
                    if cleaned:
                        parts.append(cleaned)
                elif item_type in ("image_url", "image", "video_url", "video"):
                    img_url_obj = (
                        item.get(item_type)
                        if item_type in item
                        else (
                            item.get("image_url")
                            or item.get("image")
                            or item.get("video_url")
                            or item.get("video")
                        )
                    )
                    if isinstance(img_url_obj, dict):
                        url_val = img_url_obj.get("url", "")
                    else:
                        url_val = str(img_url_obj or "")
                    if not url_val and isinstance(item.get("url"), str):
                        url_val = item["url"]
                    if url_val:
                        fpath, mtype = _materialize_media(url_val, media_sink)
                        if fpath:
                            parts.append(f"[{mtype}]({fpath})")
                        else:
                            # Fail-closed: a refused or unmaterializable
                            # reference must NOT degrade into a raw URL /
                            # path handed to agy as pseudo-media.
                            logger.debug(
                                "Dropped %s part (materialization refused): %.120s",
                                item_type,
                                url_val,
                            )
                elif item_type == "input_audio":
                    parts.append("[Audio input]")
                elif "text" in item:
                    cleaned = _clean_rendered_text(str(item.get("text") or ""))
                    if cleaned:
                        parts.append(cleaned)
        return _normalize_media_notes("\n".join(p for p in parts if p).strip(), media_sink)
    return _normalize_media_notes(_clean_rendered_text(str(content)), media_sink)


def _latest_user_request_section(user_text: str) -> str:
    """Format the canonical LATEST USER REQUEST section and anti-loop instruction."""
    return (
        f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}\n\n"
        "INSTRUCTION: Respond directly and specifically to the LATEST USER REQUEST above. "
        "Do NOT repeat previous architectural summaries, code reviews, or overview boilerplate unless explicitly asked."
        " If the LATEST USER REQUEST comments on, questions, or gives feedback about prior work rather than asking to continue it, address THAT message and do NOT silently continue the earlier task. If the user's intent is genuinely ambiguous, ask ONE short clarifying question instead of proceeding."
    )


# Core strips private message keys before providers see the request. Match only
# the complete known recovery text, never arbitrary user text containing it.
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)


def _is_recovery_nudge(message: dict[str, Any]) -> bool:
    return bool(message.get("_empty_recovery_synthetic")) or (
        _render_message_content(message.get("content")) == _EMPTY_TOOL_RESPONSE_NUDGE
    )


def _latest_user_text(messages: Sequence[Any]) -> str:
    """Return the rendered text of the last user message, or an empty string."""
    for msg in reversed(list(messages or [])):
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "user":
            if not _is_recovery_nudge(msg):
                return _render_message_content(msg.get("content"))
    return ""


def _pending_user_request_section(user_text: str) -> str:
    """Restate the user request that the current tool loop is serving.

    Tool-result turns otherwise end with a generic "continue" line, so after a
    few tool calls the request falls far behind the newest text the model sees
    and the model drifts back to the dominant pattern in the transcript (for
    example re-sending an earlier completion report).
    """
    return (
        f"### USER REQUEST THIS TOOL LOOP IS SERVING:\nUser:\n{user_text}\n\n"
        "INSTRUCTION: Use the tool results only to answer THIS request. "
        "When you have enough information, answer it directly and specifically. "
        "Do NOT repeat earlier reports or summaries from the transcript unless this request asks for them."
    )


def _tool_loop_objective_section(user_text: str) -> str:
    """Anchor the request a tool loop serves above the transcript.

    A full prompt that ends on a tool result (first call after a worker
    respawn or a broken pipe) otherwise names the request only at the tail,
    and a long transcript drowns a tail-only instruction.
    """
    return (
        "### ACTIVE OBJECTIVE:\n"
        f"The user request being served is:\n\"{user_text}\"\n\n"
        "CRITICAL: The conversation transcript below contains prior history and the tool calls made so far. "
        "Use the tool results ONLY to address the ACTIVE OBJECTIVE above. "
        "When tool execution is complete, your final answer must directly address this objective. "
        "Do NOT repeat earlier completion reports, summaries, or boilerplate from previous tasks, "
        "and do NOT resume an earlier task."
    )


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    media_sink: Optional[set] = None,
) -> str:
    """Assemble Hermes message history, system prompt, and tool schemas into a canonical prompt."""
    parts = _assemble_full_prompt_parts(messages, tools=tools, tool_choice=tool_choice, media_sink=media_sink)
    return _join_full_prompt_parts(parts)


class _FullPromptParts(NamedTuple):
    """The pieces of a full-send prompt, in the order they are joined.

    ``lead`` (preamble, tool schemas, system instructions), ``objective`` (the
    ACTIVE OBJECTIVE anchor, possibly empty), the transcript entries (joined
    under one ``Conversation transcript:`` section) and ``tail`` (the closing
    instruction and latest request).
    """

    lead: list[str]
    objective: list[str]
    transcript: list[str]
    tail: list[str]


def _join_full_prompt_parts(parts: "_FullPromptParts") -> str:
    sections = [*parts.lead, *parts.objective]
    if parts.transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(parts.transcript))
    sections.extend(parts.tail)
    return "\n\n".join(s.strip() for s in _apply_media_viewing_protocol(sections) if s and s.strip())


def _assemble_full_prompt_parts(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    media_sink: Optional[set] = None,
) -> _FullPromptParts:
    """Build the pieces of the full-send prompt (see _format_messages_as_prompt)."""
    try:
        from agent.acp_openai_bridge import render_tool_bridge_sections
        tool_sections = render_tool_bridge_sections(tools, tool_choice)
    except Exception:
        tool_sections = []
        if tools:
            tool_sections.append(
                "Available tools (OpenAI function schema):\n"
                + json.dumps(tools, ensure_ascii=False)
            )

    sections: list[str] = [*_PROMPT_PREAMBLE, *tool_sections]

    valid_messages = [m for m in messages if isinstance(m, dict)]
    last_msg = valid_messages[-1] if valid_messages else None
    last_role = str(last_msg.get("role") or "").strip().lower() if last_msg else ""

    system_parts: list[str] = []
    history_messages: list[dict[str, Any]] = []

    for msg in valid_messages:
        role = str(msg.get("role") or "unknown").strip().lower()
        if role == "system":
            rendered = _render_message_content(msg.get("content"), media_sink)
            if rendered:
                system_parts.append(rendered)
        else:
            history_messages.append(msg)

    if system_parts:
        sections.append("### SYSTEM INSTRUCTIONS (HERMES AGENT):\n" + "\n\n".join(f"System:\n{p}" for p in system_parts))

    _HISTORICAL_TOOL_RETENTION_COUNT = 8
    _HISTORICAL_TOOL_MAX_CHARS = 300
    _TOOL_MAX_CHARS_CAP = 20000

    transcript: list[str] = []
    recovery_turn = last_role == "user" and last_msg is not None and _is_recovery_nudge(last_msg)
    prior_messages = history_messages[:-1] if last_role == "user" and not recovery_turn and len(history_messages) > 1 else history_messages

    tool_indices = [
        i for i, msg in enumerate(prior_messages)
        if str(msg.get("role") or "").strip().lower() == "tool"
    ]
    recent_tool_cutoff = (
        set(tool_indices[-_HISTORICAL_TOOL_RETENTION_COUNT:])
        if len(tool_indices) > _HISTORICAL_TOOL_RETENTION_COUNT
        else set(tool_indices)
    )

    for i, message in enumerate(prior_messages):
        role = str(message.get("role") or "unknown").strip().lower()
        rendered_content = _render_message_content(message.get("content"))

        if role == "tool":
            tool_id = str(message.get("tool_call_id") or message.get("name") or "tool").strip()
            if i not in recent_tool_cutoff and len(rendered_content) > _HISTORICAL_TOOL_MAX_CHARS:
                rendered_content = (
                    rendered_content[:200]
                    + f"\n[... tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-100:]
                )
            elif len(rendered_content) > _TOOL_MAX_CHARS_CAP:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            transcript.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue

        if role == "assistant":
            parts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                parts.append(rendered_content)
            if tool_calls := message.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        parts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if parts:
                transcript.append("Assistant:\n" + "\n".join(parts))
            continue

        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            transcript.append(f"{label}:\n{rendered_content}")

    objective_sections: list[str] = []
    if recovery_turn:
        objective = _latest_user_text(history_messages)
        if objective:
            objective_sections.append(_tool_loop_objective_section(objective))
    elif last_role == "user" and last_msg is not None and len(history_messages) > 1:
        user_text = _render_message_content(last_msg.get("content"), media_sink)
        objective_sections.append(
            "### ACTIVE OBJECTIVE:\n"
            f"The user has submitted a new request:\n\"{user_text}\"\n\n"
            "CRITICAL: The conversation transcript below is historical background. "
            "Do NOT silently continue prior tool execution loops or earlier coding tasks unless explicitly requested by the ACTIVE OBJECTIVE. "
            "Focus your attention on answering the new request."
        )
    elif last_role == "tool":
        pending_objective = _latest_user_text(history_messages)
        if pending_objective:
            objective_sections.append(_tool_loop_objective_section(pending_objective))

    tail_sections: list[str] = []
    if last_role == "tool":
        tail_sections.append(
            "### LATEST TOOL RESULTS RECEIVED.\n"
            "INSTRUCTION: Evaluate the latest tool results in the transcript above and continue the task. "
            "If more tools are needed, emit <tool_call> tags. If you have enough information, "
            "respond clearly to the user without repeating prior summaries."
        )
        pending = _latest_user_text(history_messages)
        if pending:
            tail_sections.append(_pending_user_request_section(pending))
    elif last_role == "user" and last_msg is not None:
        user_text = _latest_user_text(history_messages)
        if user_text:
            tail_sections.append(_latest_user_request_section(user_text))
    else:
        tail_sections.append("Continue the conversation from the latest message.")
    return _FullPromptParts(
        lead=sections,
        objective=objective_sections,
        transcript=transcript,
        tail=tail_sections,
    )


# --- Full-send segmentation (Gemini single-message ceiling) -----------------
#
# Measured with agy 1.5.15 on a Google AI Pro account (gemini-4-neon-low): a
# single user message is silently cut and only its head is kept. 295k- and
# 826k-char prompts of ~2.1 chars/token filler both capped at 102,525 input
# tokens; a 397k-char prompt of ~1.5 chars/token filler capped at 135,143.
# Both fit a cut near ~190k characters rather than a fixed token count (an
# inference from two fillers, not a confirmed rule). The same conversation
# accepts many such messages in a row: three ~56k-token messages sent as
# consecutive turns reached input_tokens 180,204 and every head/tail code was
# recalled. A full send that does not fit one message is therefore split into
# consecutive user turns of one agy conversation. Every part but the last asks
# for a bare ACK, and those replies are consumed by the stream, never surfaced.
#
# Sizes are budgeted in "weight": one unit per ASCII character, two per
# non-ASCII character. Number-dense ASCII (inventory-style filler lines)
# measured ~1.5 chars/token on gemini-4-neon-low (396,822 chars -> 270,413
# context tokens including agy's system prompt) and CJK runs ~1 char/token,
# so 90,000 weight stays near 60k tokens for any mix, under the ~75k cut.
_GEMINI_SEGMENT_WEIGHT = 90_000
_SEGMENT_WEIGHT_ENV_VARS = (
    "HERMES_ANTIGRAVITY_SEGMENT_WEIGHT",
    "ANTIGRAVITY_SEGMENT_WEIGHT",
)
# Room kept free in every part for the part header/footer.
_SEGMENT_WRAPPER_RESERVE = 1_500
_MIN_SEGMENT_WEIGHT = 4_000
SEGMENT_ACK_TEXT = "ACK"


def _prompt_weight(text: str) -> int:
    """Conservative size proxy for token budgeting (see _GEMINI_SEGMENT_WEIGHT)."""
    if text.isascii():
        return len(text)
    return len(text) + sum(1 for ch in text if ord(ch) > 127)


def full_send_segment_budget(model: str | None) -> int | None:
    """Per-message weight budget for a full send to *model*, or None for no split.

    ``HERMES_ANTIGRAVITY_SEGMENT_WEIGHT`` (or ``ANTIGRAVITY_SEGMENT_WEIGHT``)
    overrides the budget for every model; ``0`` disables splitting. Without an
    override only Gemini routes split: Claude routes accepted >122k-token
    single messages with no truncation (issue #49).
    """
    for var in _SEGMENT_WEIGHT_ENV_VARS:
        raw = os.environ.get(var)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            continue
        if value <= 0:
            return None
        return max(value, _MIN_SEGMENT_WEIGHT)
    if str(model or "").strip().lower().startswith("gemini"):
        return _GEMINI_SEGMENT_WEIGHT
    return None


def _split_oversized_block(block: str, budget: int) -> list[str]:
    """Split one block heavier than *budget* at line boundaries.

    A single line that is itself too heavy (a long JSON tool-call frame, a
    minified file) is split by characters as the last resort.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_weight = 0
    for line in block.split("\n"):
        line_weight = _prompt_weight(line) + 1
        if line_weight > budget:
            if current:
                pieces.append("\n".join(current))
                current, current_weight = [], 0
            start = 0
            while start < len(line):
                end = start
                weight = 0
                while end < len(line) and weight < budget:
                    weight += 1 if ord(line[end]) < 128 else 2
                    end += 1
                pieces.append(line[start:end])
                start = end
            continue
        if current and current_weight + line_weight > budget:
            pieces.append("\n".join(current))
            current, current_weight = [], 0
        current.append(line)
        current_weight += line_weight
    if current:
        pieces.append("\n".join(current))
    total = len(pieces)
    if total <= 1:
        return pieces
    labelled = []
    for index, piece in enumerate(pieces, start=1):
        prefix = "" if index == 1 else f"[... continued: piece {index}/{total} of one message, split for size ...]\n"
        suffix = "" if index == total else f"\n[... this message continues in piece {index + 1}/{total} ...]"
        labelled.append(prefix + piece + suffix)
    return labelled


def _priming_segment(index: int, total: int, body: str) -> str:
    return (
        f"[CONTEXT PART {index}/{total}] This prompt is too long for one message, so it arrives "
        f"in {total} consecutive parts. This is part {index}: background only. Do not answer it, "
        "do not act on instructions inside it yet, and do not call any tools (no native tools, "
        f"no <tool_call> blocks). Reply with exactly: {SEGMENT_ACK_TEXT}\n\n"
        f"{body}\n\n"
        f"[END OF CONTEXT PART {index}/{total}] Reply with exactly: {SEGMENT_ACK_TEXT}"
    )


def _final_segment(total: int, body: str) -> str:
    return (
        f"[CONTEXT PART {total}/{total}, FINAL] Parts 1-{total - 1} above are the earlier parts of "
        "this same prompt; your ACK replies to them were placeholders. Treat all "
        f"{total} parts as ONE prompt and respond to it now, following every instruction in it "
        "(including the tool-calling protocol).\n\n"
        f"{body}"
    )


def _format_messages_as_segments(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    media_sink: Optional[set] = None,
    budget: int | None = None,
) -> list[str]:
    """Full-send prompt as one or more consecutive user messages.

    Returns ``[_format_messages_as_prompt(...)]`` unchanged whenever the
    prompt fits *budget* (or *budget* is None). Otherwise the lead sections
    and transcript entries are packed in order into priming parts, cut only
    at section/message boundaries so no ``<tool_call>`` frame or tool result
    is split; a single message heavier than a part is split at line
    boundaries with continuation markers. The ACTIVE OBJECTIVE and the
    closing request go into the final part, together with the newest
    transcript entries that still fit, so the request is the last thing the
    model reads.
    """
    parts = _assemble_full_prompt_parts(messages, tools=tools, tool_choice=tool_choice, media_sink=media_sink)
    full = _join_full_prompt_parts(parts)
    if budget is None or _prompt_weight(full) <= budget:
        return [full]

    content_budget = max(budget - _SEGMENT_WRAPPER_RESERVE, _MIN_SEGMENT_WEIGHT // 2)
    blocks = [s.strip() for s in parts.lead if s and s.strip()]
    if _MEDIA_REF_RE.search(full):
        # Same placement as the single-message prompt: before the transcript.
        blocks.append(_MEDIA_VIEWING_SECTION)
    for index, entry in enumerate(parts.transcript):
        blocks.append(("Conversation transcript:\n\n" if index == 0 else "") + entry)
    final_sections = [s.strip() for s in (*parts.objective, *parts.tail) if s and s.strip()]
    final_block = "\n\n".join(final_sections)

    # Pieces never exceed content_budget; a heavy final block is split too, and
    # only its last piece stays in the final part.
    pieces: list[str] = []
    for block in blocks:
        if _prompt_weight(block) > content_budget:
            pieces.extend(_split_oversized_block(block, content_budget))
        else:
            pieces.append(block)
    final_pieces = (
        _split_oversized_block(final_block, content_budget)
        if _prompt_weight(final_block) > content_budget
        else [final_block]
    )
    pieces.extend(final_pieces[:-1])
    final_body = final_pieces[-1]

    # The final part takes the newest pieces that still fit next to the request.
    final_weight = _prompt_weight(final_body)
    tail_pieces: list[str] = []
    while pieces and final_weight + _prompt_weight(pieces[-1]) + 2 <= content_budget:
        final_weight += _prompt_weight(pieces[-1]) + 2
        tail_pieces.insert(0, pieces.pop())

    chunks: list[list[str]] = []
    current: list[str] = []
    current_weight = 0
    for piece in pieces:
        weight = _prompt_weight(piece) + 2
        if current and current_weight + weight > content_budget:
            chunks.append(current)
            current, current_weight = [], 0
        current.append(piece)
        current_weight += weight
    if current:
        chunks.append(current)
    if not chunks:
        return [full]

    total = len(chunks) + 1
    segments = [_priming_segment(i, total, "\n\n".join(chunk)) for i, chunk in enumerate(chunks, start=1)]
    segments.append(_final_segment(total, "\n\n".join([*tail_pieces, final_body])))
    return segments


def _messages_match_prefix(history: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]]) -> bool:
    """Return True if incoming messages strictly extend history as a continuation."""
    if not history or len(incoming) <= len(history):
        return False
    for i, h_msg in enumerate(history):
        inc_msg = incoming[i]
        if not isinstance(inc_msg, dict) or not isinstance(h_msg, dict):
            return False
        if inc_msg.get("role") != h_msg.get("role"):
            return False
        if inc_msg.get("content") != h_msg.get("content"):
            return False
        if inc_msg.get("tool_calls") != h_msg.get("tool_calls"):
            return False
    return True


def _format_delta_prompt(
    new_messages: Sequence[dict[str, Any]],
    pending_user_text: str = "",
    media_sink: Optional[set] = None,
) -> str:
    """Format only the incremental messages in an ongoing multi-turn interaction.

    ``pending_user_text`` is the latest user request from the full history.
    A tool-result delta does not contain it, so the caller passes it in and it
    is restated after the continue instruction.
    """
    valid_messages = [m for m in new_messages if isinstance(m, dict)]
    if not valid_messages:
        return ""

    last_msg = valid_messages[-1]
    last_role = str(last_msg.get("role") or "").strip().lower()

    recovery_turn = last_role == "user" and _is_recovery_nudge(last_msg)
    if last_role == "user" and not recovery_turn:
        prior_messages = valid_messages[:-1]
    else:
        prior_messages = valid_messages

    parts: list[str] = []
    for msg in prior_messages:
        role = str(msg.get("role") or "").strip().lower()
        rendered_content = _render_message_content(msg.get("content"), media_sink)
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("name") or "tool").strip()
            if len(rendered_content) > 20000:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            parts.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue
        if role == "assistant":
            subparts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                subparts.append(rendered_content)
            if tool_calls := msg.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        subparts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if subparts:
                parts.append("Assistant:\n" + "\n".join(subparts))
            continue
        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            parts.append(f"{label}:\n{rendered_content}")

    if last_role == "tool":
        parts.append(
            "Continue the conversation from the latest tool result. "
            "REMINDER: You are in HEADLESS INFERENCE MODE. Native function-calling and native tools are disabled. "
            "If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in plain text with valid JSON and a properly JSON-escaped 'arguments' string. "
            "Never invoke native tools or emit native functionCall blocks."
        )
        if pending_user_text:
            parts.append(_pending_user_request_section(pending_user_text))
    elif last_role == "user":
        user_text = (pending_user_text or _latest_user_text(valid_messages)) if recovery_turn else _render_message_content(last_msg.get("content"), media_sink)
        if user_text:
            parts.append(_latest_user_request_section(user_text))
    else:
        parts.append("Continue the conversation from the latest message above.")
    return "\n\n".join(s.strip() for s in _apply_media_viewing_protocol(parts) if s and s.strip())


def media_reference_paths(messages: Any, media_sink: Optional[set] = None) -> set[str]:
    """Every media path the rendered messages reference: ``[Image](/abs/path)``.

    Shared by the stream's native-tool allowlist and the client's worker
    permissions stamp, so both describe the same media contract. Rendering
    materializes references (downloads, cache files) into *media_sink* when
    given, so the caller can clean them up at session close.
    """
    paths: set[str] = set()
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        rendered = _render_message_content(message.get("content"), media_sink)
        for path in _MEDIA_REF_RE.findall(rendered):
            if path:
                paths.add(os.path.normpath(path))
    return paths


def _longest_tool_call_prefix_match(text: str) -> int:
    """Return the length of the longest suffix of `text` that matches a prefix of `<tool_call>`."""
    for prefix in _TOOL_CALL_PREFIXES:
        if text.endswith(prefix):
            return len(prefix)
    return 0


def _has_unfinished_tool_call_frame(
    text: str, state: _MarkdownCodeState | None = None
) -> bool:
    """True when *text* ends inside a ``<tool_call>`` frame.

    Uses the incremental splitter's frame boundary (:func:`_find_tool_call_close`)
    so a complete block whose JSON contains a literal closing tag is not
    mistaken for a truncated one. An opening tag without a matching close, or
    a trailing fragment of the opening tag (``<tool_cal``), is unfinished.
    Tags inside markdown code spans or fenced code blocks are ignored.
    """
    cur = state.clone() if state is not None else _MarkdownCodeState()
    cursor = 0
    while True:
        rel_start = _find_unquoted_tool_call_open(text[cursor:], cur)
        if rel_start == -1:
            k = _longest_tool_call_prefix_match(text[cursor:])
            if k <= 1:
                return False
            cur.advance(text[cursor : len(text) - k])
            return not cur.in_code()
        start = cursor + rel_start
        if not text.startswith(_TOOL_CALL_OPEN, start):
            if start + len("<tool_call") == len(text):
                return True
            cur.advance(text[cursor : start + len("<tool_call")])
            cursor = start + len("<tool_call")
            continue
        close = _find_tool_call_close(text, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            return True
        end = close + len(_TOOL_CALL_CLOSE)
        cur.advance(text[cursor:end])
        cursor = end


_JSON_QUOTE_FIX_ERRORS = ("Expecting ',' delimiter", "Expecting ':' delimiter", "Extra data")
_MAX_JSON_FIXES = 200
# A quoted key right after the quote the decoder stopped at.
_MISSING_MEMBER_COMMA_RE = re.compile(r'\s*"[^"\\]*"\s*:')


def _lenient_json_loads(text: str) -> tuple[Any, dict[str, int]]:
    """Decode JSON a model wrote by hand, fixing two escaping slips.

    Shell commands and Python snippets inside a ``<tool_call>`` defeat models
    that must escape them for JSON, and for a string-valued ``arguments``
    twice. Observed on gemini-4-neon (agy 1.5.15), every time with valid JSON
    around the slip:

    * a backslash before a character JSON cannot escape (``\\(`` ``\\|``
      ``\\.`` from ``find \\( ... \\)`` or ``grep 'a\\|b'``). JSON gives such a
      sequence no meaning, so the only reading is the literal backslash the
      shell needs: it is doubled. The exception is ``\\'``: JSON does not
      escape single quotes, so it is a plain ``'`` written by habit and the
      backslash is dropped.
    * a bare ``"`` inside a string (``f'{"DIR" if d else "FILE"}'``). The
      decoder ends the string there and then fails on the next character.
      The quote that ended the string is escaped and decoding resumes.

    Each fix is driven by the decoder's own error position and touches one
    backslash (inserted, or dropped for ``\\'``); every other character the
    model wrote is kept, and the result must still decode cleanly. Returns ``(value, fixes)`` and raises
    ``ValueError`` when the text cannot be decoded this way. Callers that can
    check the result against a schema must do so: a quote fix can merge two
    keys a missing comma separated.
    """
    fixes = {"escape": 0, "quote": 0}
    s = text
    # A slip is one character, so a body can carry at most len(text) of
    # them; a generated grep pipeline runs to one every dozen. The budget
    # bounds the quadratic loop (one decode attempt per fix) without
    # refusing a long hand-escaped script for its number of slips.
    budget = max(_MAX_JSON_FIXES, len(text) // 10)
    for _ in range(budget + 1):
        try:
            return json.loads(s, strict=False), fixes
        except json.JSONDecodeError as exc:
            pos = exc.pos
            if exc.msg.startswith(("Invalid \\escape", "Invalid \\uXXXX")):
                slash = pos if pos < len(s) and s[pos] == "\\" else s.rfind("\\", 0, pos + 1)
                if slash < 0:
                    raise ValueError(f"{exc.msg} (char {pos})") from None
                if s.startswith("\\'", slash):
                    # JSON never escapes a single quote; "\'" is the habit of
                    # languages that do, and means a plain quote. Doubling it
                    # would hand bash a literal backslash-quote: sed 's/a/b/'
                    # would become sed \'s/a/b/\'.
                    s = s[:slash] + s[slash + 1:]
                else:
                    s = s[:slash] + "\\" + s[slash:]
                fixes["escape"] += 1
                continue
            if exc.msg in _JSON_QUOTE_FIX_ERRORS:
                quote = s.rfind('"', 0, pos)
                if quote < 0 or s[quote + 1:pos].strip(" \t\r\n}]"):
                    raise ValueError(f"{exc.msg} (char {pos})") from None
                slashes = 0
                while quote - slashes - 1 >= 0 and s[quote - slashes - 1] == "\\":
                    slashes += 1
                if slashes % 2:
                    raise ValueError(f"{exc.msg} (char {pos})") from None
                if _MISSING_MEMBER_COMMA_RE.match(s, quote + 1):
                    # A quoted key follows: the slip is a missing comma between
                    # two members, not a bare quote inside a string. Escaping
                    # would merge the key into the previous value, and the
                    # schema check cannot see it when that key was optional
                    # ({"path": "/x" "content": "hi"} wrote to /x" "content": "hi).
                    raise ValueError(f"missing ',' between members (char {quote + 1})") from None
                s = s[:quote] + "\\" + s[quote:]
                fixes["quote"] += 1
                continue
            raise ValueError(f"{exc.msg} (char {pos})") from None
    raise ValueError("too many JSON repairs")


def _arguments_fit_schema(args: dict[str, Any], schema: Any) -> str | None:
    """Why repaired *args* do not fit a tool's JSON schema, or None when they do.

    Only applied to arguments that needed a repair: a quote fix that merged
    two keys loses one, which shows up as a missing required key or an
    unknown key.
    """
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    required = schema.get("required") or []
    missing = [key for key in required if isinstance(key, str) and key not in args]
    if missing:
        return f"missing required {', '.join(missing)}"
    if isinstance(props, dict) and schema.get("additionalProperties") is not True:
        unknown = [key for key in args if key not in props]
        if unknown:
            return f"unknown key {', '.join(unknown)}"
    return None


def _normalize_tool_arguments(
    fn_name: str, fn_args: str, tool_schemas: dict[str, Any] | None
) -> tuple[str, str | None]:
    """Validate a string ``arguments`` value; repair it when the model slipped.

    Returns ``(arguments, error)``. Valid JSON passes through byte-identical.
    Invalid JSON is repaired by :func:`_lenient_json_loads` and re-serialized;
    when that fails, or the repaired object does not fit the tool's schema,
    ``error`` says why and the call must not be executed. Hermes would
    otherwise get the broken string, fail to repair it, and treat the turn as
    a truncated reply: retried with the same mistake, then failed.
    """
    stripped = fn_args.strip()
    if not stripped:
        return "{}", None
    try:
        json.loads(stripped)
        return fn_args, None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        value, fixes = _lenient_json_loads(stripped)
    except ValueError as exc:
        return fn_args, f"arguments are not valid JSON ({exc})"
    if not isinstance(value, dict):
        return fn_args, "arguments must be a JSON object"
    schema = (tool_schemas or {}).get(fn_name)
    misfit = _arguments_fit_schema(value, schema)
    if misfit:
        return fn_args, f"arguments are not valid JSON; the repaired object does not fit the schema ({misfit})"
    logger.warning(
        "Repaired <tool_call> arguments for %s: %d invalid escape(s), %d unescaped quote(s)",
        fn_name,
        fixes["escape"],
        fixes["quote"],
    )
    return json.dumps(value, ensure_ascii=False), None


def _tool_call_from_object(
    obj: Any,
    ordinal: int,
    tool_schemas: dict[str, Any] | None = None,
    *,
    frame_repaired: bool = False,
) -> Any | None:
    """Build one tool-call delta object from a decoded JSON value.

    Mirrors the core ACP parser: the value must be an object whose
    ``function`` is an object with a non-blank string name. Anything else is
    malformed and returns None, so a nameless call is never emitted.

    A string ``arguments`` is validated (see :func:`_normalize_tool_arguments`).
    A call whose arguments cannot be made valid carries ``arguments_error``;
    the stream withholds it from Hermes and asks the model to re-emit it.
    """
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if not isinstance(fn, dict):
        return None
    fn_name = fn.get("name")
    if not isinstance(fn_name, str) or not fn_name.strip():
        return None
    fn_name = fn_name.strip()
    raw_call_id = obj.get("id")
    id_synthesized = not isinstance(raw_call_id, str) or not raw_call_id.strip()
    call_id = f"call_{ordinal}" if id_synthesized else raw_call_id
    raw_args = fn.get("arguments", "{}")
    arguments_error = None
    args_repaired = False
    if isinstance(raw_args, str):
        fn_args, arguments_error = _normalize_tool_arguments(fn_name, raw_args, tool_schemas)
        args_repaired = fn_args != raw_args
    else:
        fn_args = json.dumps(raw_args, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=fn_name, arguments=fn_args),
        arguments_error=arguments_error,
        protocol_mutated=bool(frame_repaired or id_synthesized or args_repaired),
    )


def _has_closed_tool_call_frame(
    text: str, state: _MarkdownCodeState | None = None
) -> bool:
    """True when *text* contains a closed ``<tool_call>...</tool_call>`` frame."""
    cur = state.clone() if state is not None else _MarkdownCodeState()
    cursor = 0
    while True:
        rel_start = _find_unquoted_tool_call_open(text[cursor:], cur)
        if rel_start == -1:
            return False
        start = cursor + rel_start
        if not text.startswith(_TOOL_CALL_OPEN, start):
            cur.advance(text[cursor : start + len("<tool_call")])
            cursor = start + len("<tool_call")
            continue
        return _find_tool_call_close(text, start + len(_TOOL_CALL_OPEN)) != -1


def _has_unparseable_tool_call_frame(
    text: str, state: _MarkdownCodeState | None = None, tool_schemas: dict[str, Any] | None = None
) -> bool:
    """True when *text* keeps a closed ``<tool_call>`` frame after parsing.

    :func:`_parse_tool_block` consumes every frame it can turn into a call and
    leaves the unparseable ones in its cleaned text: invalid JSON, a missing
    or blank ``function`` name, a non-object value, or an empty body. A
    closed frame surviving in that cleaned text is therefore an explicit
    protocol frame with no call. Unclosed frames are
    :func:`_has_unfinished_tool_call_frame`'s concern.
    """
    _calls, cleaned = _parse_tool_block(text, state=state, tool_schemas=tool_schemas)
    return _has_closed_tool_call_frame(cleaned, state=state)


def _extract_tagged_tool_calls(
    block: str, state: _MarkdownCodeState | None = None, tool_schemas: dict[str, Any] | None = None
) -> tuple[list[Any], str]:
    """Extract ``<tool_call>{...}</tool_call>`` frames using the shared scanner.

    The frame boundary is :func:`_find_tool_call_close`, so a literal closing
    tag inside a JSON string does not terminate the block. Text outside valid
    frames, and a frame whose body is not a valid call, are returned as
    ``cleaned_text``. Tags inside markdown code spans or fenced code blocks
    are left untouched as ordinary text.
    """
    calls: list[Any] = []
    text_parts: list[str] = []
    cur = state.clone() if state is not None else _MarkdownCodeState()
    cursor = 0
    while True:
        rel_start = _find_unquoted_tool_call_open(block[cursor:], cur)
        if rel_start == -1:
            text_parts.append(block[cursor:])
            break
        start = cursor + rel_start
        if not block.startswith(_TOOL_CALL_OPEN, start):
            text_parts.append(block[cursor : start + len("<tool_call")])
            cur.advance(block[cursor : start + len("<tool_call")])
            cursor = start + len("<tool_call")
            continue
        text_parts.append(block[cursor:start])
        cur.advance(block[cursor:start])
        close = _find_tool_call_close(block, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            text_parts.append(block[start:])
            break
        end = close + len(_TOOL_CALL_CLOSE)
        body = block[start + len(_TOOL_CALL_OPEN):close].strip()
        parsed = None
        frame_repaired = False
        try:
            parsed = json.loads(body)
        except Exception:
            # Same escaping slips as in arguments (see _lenient_json_loads),
            # one level up: an object-valued "arguments" carries the command.
            try:
                parsed, fixes = _lenient_json_loads(body)
                frame_repaired = True
                logger.warning(
                    "Repaired <tool_call> frame JSON: %d invalid escape(s), %d unescaped quote(s)",
                    fixes["escape"],
                    fixes["quote"],
                )
            except ValueError:
                parsed = None
        call = _tool_call_from_object(
            parsed, len(calls) + 1, tool_schemas, frame_repaired=frame_repaired
        )
        if call is None:
            text_parts.append(block[start:end])
            cur.advance(block[start:end])
        else:
            calls.append(call)
        cursor = end
    if not calls:
        # No frame was consumed: hand the input back untouched. Joining the
        # pieces would insert separators at every tag boundary and corrupt
        # ordinary prose that merely mentions the markup.
        return [], block.strip()
    return calls, "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()


def _parse_tool_block(
    block: str, state: _MarkdownCodeState | None = None, tool_schemas: dict[str, Any] | None = None
) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text).

    A delta whose ``arguments_error`` is set must not be executed (see
    :func:`_tool_call_from_object`).
    """
    calls: list[Any] = []
    cleaned_text = ""
    if _TOOL_CALL_OPEN in block:
        calls, cleaned_text = _extract_tagged_tool_calls(block, state=state, tool_schemas=tool_schemas)
    else:
        try:
            from agent.acp_openai_bridge import extract_tool_calls_from_text
            calls, cleaned_text = extract_tool_calls_from_text(block)
        except Exception:
            cleaned_text = block

    deltas = []
    for i, call in enumerate(calls):
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args = getattr(fn, "arguments", "{}") if fn else "{}"
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        deltas.append(
            SimpleNamespace(
                index=i,
                id=getattr(call, "id", f"call_{i+1}"),
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
                arguments_error=getattr(call, "arguments_error", None),
                protocol_mutated=bool(getattr(call, "protocol_mutated", False)),
            )
        )
    return deltas, cleaned_text.strip()
