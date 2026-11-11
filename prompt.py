"""Prompt construction, message translation, and tool-call parsing for Antigravity."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Optional

try:
    from hermes_constants import get_hermes_home
except ImportError:  # standalone use (tests, plugin dev)
    def get_hermes_home():
        return os.path.join(os.path.expanduser("~"), ".hermes")

logger = logging.getLogger(__name__)

# Media materialized for agy lives under the active Hermes home (profile-aware)
# and is deleted when the owning client session closes.
_MEDIA_CACHE_DIR = os.path.join(get_hermes_home(), "cache", "agy_media")
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


def _register_materialized(
    fpath: str, mtype: str, media_sink: Optional[set]
) -> tuple[str, str]:
    if fpath and media_sink is not None:
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
    os.makedirs(_MEDIA_CACHE_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=_MEDIA_CACHE_DIR, suffix=".part")
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
        fpath = os.path.join(_MEDIA_CACHE_DIR, f"{digest.hexdigest()[:16]}{ext}")
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
            fpath = os.path.join(_MEDIA_CACHE_DIR, f"{h}{ext}")
            if not os.path.exists(fpath):
                try:
                    os.makedirs(_MEDIA_CACHE_DIR, exist_ok=True)
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
        return _register_materialized(os.path.abspath(local), kind, media_sink)

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
        return _register_materialized(os.path.abspath(url), kind, media_sink)

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
    "You are operating in HEADLESS INFERENCE MODE. In this mode, your native tools (run_command, write_to_file, view_file, edit_file, etc.) are disabled and will be silently rejected. Do not attempt to use them.",
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent on the host side.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output.",
    "- Each tool call must be a JSON object containing 'id', 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}.",
    "- 'arguments' must be a JSON-encoded string containing the function arguments.",
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

    if recovery_turn:
        objective = _latest_user_text(history_messages)
        if objective:
            sections.append(_tool_loop_objective_section(objective))
    elif last_role == "user" and last_msg is not None and len(history_messages) > 1:
        user_text = _render_message_content(last_msg.get("content"), media_sink)
        sections.append(
            "### ACTIVE OBJECTIVE:\n"
            f"The user has submitted a new request:\n\"{user_text}\"\n\n"
            "CRITICAL: The conversation transcript below is historical background. "
            "Do NOT silently continue prior tool execution loops or earlier coding tasks unless explicitly requested by the ACTIVE OBJECTIVE. "
            "Focus your attention on answering the new request."
        )
    elif last_role == "tool":
        pending_objective = _latest_user_text(history_messages)
        if pending_objective:
            sections.append(_tool_loop_objective_section(pending_objective))

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    if last_role == "tool":
        sections.append(
            "### LATEST TOOL RESULTS RECEIVED.\n"
            "INSTRUCTION: Evaluate the latest tool results in the transcript above and continue the task. "
            "If more tools are needed, emit <tool_call> tags. If you have enough information, "
            "respond clearly to the user without repeating prior summaries."
        )
        pending = _latest_user_text(history_messages)
        if pending:
            sections.append(_pending_user_request_section(pending))
    elif last_role == "user" and last_msg is not None:
        user_text = _latest_user_text(history_messages)
        if user_text:
            sections.append(_latest_user_request_section(user_text))
    else:
        sections.append("Continue the conversation from the latest message.")
    return "\n\n".join(s.strip() for s in _apply_media_viewing_protocol(sections) if s and s.strip())


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
            "REMINDER: You are in HEADLESS INFERENCE MODE. Native tools are disabled. "
            "If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output. "
            "Do NOT invoke native tools directly."
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


def _has_unfinished_tool_call_frame(text: str) -> bool:
    """True when *text* ends inside a ``<tool_call>`` frame.

    Uses the incremental splitter's frame boundary (:func:`_find_tool_call_close`)
    so a complete block whose JSON contains a literal closing tag is not
    mistaken for a truncated one. An opening tag without a matching close, or
    a trailing fragment of the opening tag (``<tool_cal``), is unfinished.
    """
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            # A lone "<" is ordinary text (a comparison operator, an unfinished
            # HTML tag, ...), not evidence of a truncated <tool_call> frame.
            # Two or more matching characters ("<t" ... "<tool_call") remain
            # fail-closed: the shortest useful proof of protocol framing.
            return _longest_tool_call_prefix_match(text) > 1
        close = _find_tool_call_close(text, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            return True
        cursor = close + len(_TOOL_CALL_CLOSE)


def _tool_call_from_object(obj: Any, ordinal: int) -> Any | None:
    """Build one tool-call delta object from a decoded JSON value.

    Mirrors the core ACP parser: the value must be an object whose
    ``function`` is an object with a non-blank string name. Anything else is
    malformed and returns None, so a nameless call is never emitted.
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
    call_id = obj.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call_{ordinal}"
    fn_args = fn.get("arguments", "{}")
    if not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=fn_name, arguments=fn_args),
    )


def _has_unparseable_tool_call_frame(text: str) -> bool:
    """True when *text* keeps a closed ``<tool_call>`` frame after parsing.

    :func:`_parse_tool_block` consumes every frame it can turn into a call and
    leaves the unparseable ones in its cleaned text: invalid JSON, a missing
    or blank ``function`` name, a non-object value, or an empty body. A
    closed frame surviving in that cleaned text is therefore an explicit
    protocol frame with no call, which must fail the turn instead of being
    published as assistant text. Unclosed frames are
    :func:`_has_unfinished_tool_call_frame`'s concern.
    """
    _calls, cleaned = _parse_tool_block(text)
    start = cleaned.find(_TOOL_CALL_OPEN)
    return (
        start != -1
        and _find_tool_call_close(cleaned, start + len(_TOOL_CALL_OPEN)) != -1
    )


def _extract_tagged_tool_calls(block: str) -> tuple[list[Any], str]:
    """Extract ``<tool_call>{...}</tool_call>`` frames using the shared scanner.

    The frame boundary is :func:`_find_tool_call_close`, so a literal closing
    tag inside a JSON string does not terminate the block. Text outside valid
    frames, and a frame whose body is not a valid call, are returned as
    ``cleaned_text``.
    """
    calls: list[Any] = []
    text_parts: list[str] = []
    cursor = 0
    while True:
        start = block.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            text_parts.append(block[cursor:])
            break
        text_parts.append(block[cursor:start])
        close = _find_tool_call_close(block, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            text_parts.append(block[start:])
            break
        end = close + len(_TOOL_CALL_CLOSE)
        body = block[start + len(_TOOL_CALL_OPEN):close].strip()
        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        call = _tool_call_from_object(parsed, len(calls) + 1)
        if call is None:
            text_parts.append(block[start:end])
        else:
            calls.append(call)
        cursor = end
    if not calls:
        # No frame was consumed: hand the input back untouched. Joining the
        # pieces would insert separators at every tag boundary and corrupt
        # ordinary prose that merely mentions the markup.
        return [], block.strip()
    return calls, "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()


def _parse_tool_block(block: str) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text)."""
    calls: list[Any] = []
    cleaned_text = ""
    if _TOOL_CALL_OPEN in block:
        calls, cleaned_text = _extract_tagged_tool_calls(block)
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
            )
        )
    return deltas, cleaned_text.strip()
