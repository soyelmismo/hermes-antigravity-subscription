"""Antigravity Subscription DirectSDK client for Hermes Agent.

Drives the official Antigravity CLI (`agy`) as an external process in
stream-json mode to provide request-scoped completions using the user's
existing Antigravity / Gemini subscription quota.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# prompt.py imports agent.acp_openai_bridge lazily during the first turn, while Hermes
# starts the title-generation thread in parallel. Two threads first-importing
# openai.types.chat race and the loser sees a partially initialized module, so the
# session title is lost. Importing the bridge here runs that import on the loading thread.
with contextlib.suppress(Exception):
    import agent.acp_openai_bridge  # noqa: E402,F401

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from .process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        is_authenticated,
        read_media_rules,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from .prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _latest_user_text,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
        media_reference_paths,
    )
    from .delivery import DeliveryWatch
    from .stream import AntigravityStream, collect_stream_completion
    from .debug import dump_prompt_debug
except ImportError:
    from models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        is_authenticated,
        read_media_rules,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _latest_user_text,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
        media_reference_paths,
    )
    from delivery import DeliveryWatch
    from stream import AntigravityStream, collect_stream_completion
    from debug import dump_prompt_debug

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300.0

# agy prints these when its token source cannot be built — the signature of
# OAuth contention with ANOTHER agy session on the same Google account (each
# process refreshes the shared token; the loser logs this and produces no
# result). Observed verbatim in isolated-home cli logs, 2026-11-11.
_AUTH_CONTENTION_SIGNATURES = (
    "not logged into antigravity",
    "error getting token source",
)


def _empty_no_result(completion: Any) -> bool:
    """True when a completion carries neither content, tool calls, nor a
    finish reason — the shape ``collect_stream_completion`` returns when the
    worker ended its turn without ever emitting a result event."""
    try:
        choice = completion.choices[0]
    except (AttributeError, IndexError, TypeError):
        return False
    message = getattr(choice, "message", None)
    if getattr(message, "content", None):
        return False
    if getattr(message, "tool_calls", None):
        return False
    return getattr(choice, "finish_reason", None) is None


def _stderr_shows_auth_contention(proc: Any) -> bool:
    drainer = getattr(proc, "_stderr_drainer", None)
    if drainer is None:
        return False
    try:
        tail = (drainer.get_tail() or "").lower()
    except Exception:
        return False
    return any(signature in tail for signature in _AUTH_CONTENTION_SIGNATURES)

# The persistent worker is only ever replaced by the next request (model
# change, non-continuation history) or by close(). Hermes retires an evicted
# agent's client without calling close(), so without an idle bound a finished
# conversation leaves its agy process (~200 MB RSS) running for the lifetime
# of the gateway. Terminating an idle worker is safe: it clears the recorded
# history, so the next turn takes the full-prompt branch on a fresh worker.
# ANTIGRAVITY_WORKER_IDLE_SECONDS overrides the bound; 0 or less disables it.
_DEFAULT_WORKER_IDLE_SECONDS = 900.0

# A turn abandoned mid-stream keeps _worker_lock held: the stream's generator is
# reclaimable only by the cyclic GC, so its finally block -- the only place that
# releases the lock -- can be postponed indefinitely by a lingering reference.
# The countdown then re-arms forever, which is how three dashboard workers were
# measured still alive past the bound (November 2026) while the gateway reaped
# Such a turn is dead by then -- but silence alone is not proof: a long
# reasoning turn or a slow first token can emit nothing for minutes while the
# worker is busy, and a caller may set a read deadline well past the default.
# So the second opinion needs BOTH no output for the abandon grace AND no CPU
# burned since the last output (_worker_cpu_seconds). An abandoned worker sits
# at 0% -- measured at 2.5 s CPU across sixteen minutes of life -- whereas a
# live reasoning turn does not. Where CPU time cannot be read (non-Linux, or a
# /proc entry that has gone) the worker is left alone: refusing to kill is
# always the safe failure here.
# ANTIGRAVITY_WORKER_ABANDON_SECONDS overrides the grace verbatim; 0 or less
# disables the second opinion and restores the old re-arm-forever behaviour.
# The default grace is one idle window, floored at one read deadline plus a
# minute, so it lands at ~15 minutes -- inside the external reaper's fence,
# which is what keeps that backstop silent.
_WORKER_ABANDON_GRACE_WINDOWS = 1.0
_WORKER_ABANDON_MIN_SECONDS = _DEFAULT_TIMEOUT_SECONDS + 60.0
# CPU seconds a worker may burn between signs of life before it counts as busy
# rather than abandoned. agy's own start-up costs a few tenths of a second, so
# this clears a worker that has barely started but not one that is thinking.
_WORKER_ABANDON_CPU_SECONDS = 1.0


def _worker_cpu_seconds(proc: subprocess.Popen) -> float | None:
    """Total CPU seconds the worker has consumed, or None when unreadable.

    Linux only: /proc/<pid>/stat carries utime and stime as fields 14 and 15,
    measured from *after* the comm field, which is parenthesised and may itself
    contain spaces and parentheses -- hence the split on the last ')'.
    """
    pid = getattr(proc, "pid", None)
    if not pid or not os.path.isdir("/proc"):
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read().decode("ascii", "replace")
        fields = data[data.rindex(")") + 2:].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(fields[11]) + int(fields[12])) / ticks
    except (OSError, ValueError, IndexError):
        return None


def _worker_idle_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_WORKER_IDLE_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_WORKER_IDLE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_WORKER_IDLE_SECONDS


def _worker_abandon_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_WORKER_ABANDON_SECONDS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    # Never shorter than one read deadline plus a margin, so a live turn that is
    # merely slow to speak cannot be mistaken for an abandoned one.
    return max(
        _WORKER_ABANDON_GRACE_WINDOWS * _worker_idle_seconds(),
        _WORKER_ABANDON_MIN_SECONDS,
    )

# Issue #4 (Windows, WinError 32): terminating agy's process tree
# (taskkill /F /T) is asynchronous, so a child -- or a grandchild -- can
# still hold conversations/*.db for a short moment after termination
# returns, and TemporaryDirectory.cleanup() then raises PermissionError.
# Three quick attempts (0.1s, then 0.25s apart) cover that handle-release
# window: the worst case adds ~0.35s to a close() that was failing anyway,
# while the normal case (POSIX, or Windows where the handle is already
# gone) succeeds on the first attempt and pays none of it. After the last
# attempt a forced removal drops whatever is still unlocked, so close()
# never raises and never leaves the workspace behind needlessly.
#
# The tuple is the single source of truth for the retry budget: the
# attempts count is DERIVED from it (one try plus one backoff per retry),
# so extending the budget can never desynchronize the two -- a longer
# budget than tuple would index past its end and the IndexError would be
# swallowed by _remove_temp_dir's outer guard, turning a retry budget
# into a silent workspace leak.
_TEMP_DIR_CLEANUP_BACKOFF_SECONDS = (0.1, 0.25)
_TEMP_DIR_CLEANUP_ATTEMPTS = len(_TEMP_DIR_CLEANUP_BACKOFF_SECONDS) + 1

__all__ = [
    "AGY_MARKER_BASE_URL",
    "AntigravityClient",
    "AntigravityStream",
    "_FALLBACK_MODELS",
    "_MODEL_ALIASES",
    "_PROMPT_PREAMBLE",
    "_ROLE_LABELS",
    "_TEMP_DIR_CLEANUP_ATTEMPTS",
    "_TEMP_DIR_CLEANUP_BACKOFF_SECONDS",
    "_force_rmtree",
    "_format_delta_prompt",
    "_latest_user_text",
    "_format_messages_as_prompt",
    "_kill_process_tree",
    "_longest_tool_call_prefix_match",
    "_messages_match_prefix",
    "_normalize_effort",
    "_own_process_group",
    "_parse_tool_block",
    "_render_message_content",
    "is_authenticated",
    "resolve_agy_command",
    "resolve_model_and_effort",
]


def _force_rmtree(path: str) -> None:
    """Best-effort forced removal of a workspace; never raises (issue #4).

    ``shutil.rmtree(path, ignore_errors=True)`` alone is WEAKER than the
    ``tempfile._rmtree`` that ``TemporaryDirectory.cleanup()`` uses:
    the latter chmod-resets read-only subtrees and retries them, the
    former silently skips them, which is exactly the residual leak this
    issue is about (a read-only ``conversations`` tree surviving the
    forced pass on the Windows shape of the bug). So: forced removal
    first; if anything survives, grant the owner write (and, for
    directories, search) permission -- the POSIX analogue of "the handle
    is gone now, try again" -- and sweep once more.

    The permission sweep deliberately SKIPS links using ``os.lstat`` so
    nothing is ever followed:

    * ``setup_isolated_home`` links the user's REAL OAuth token into the
      isolated HOME with ``os.symlink``, ``os.link``, or a ``copy2``
      fallback (process.py). A surviving token link is precisely the
      residual-leak shape of #4, and following it would chmod the user's
      real 0400 token OUTSIDE the workspace to 0600 -- out-of-scope
      permission widening on user data, on the exact path this issue is
      about. The residual link is a documented lesser evil. The
      hardlink case (the ``os.link`` fallback, ``st_nlink > 1``) is
      skipped for the same reason -- that inode is the user's token.
      On Windows the skip is deliberate for an additional reason:
      DeleteFile fails with ACCESS_DENIED on a file carrying
      FILE_ATTRIBUTE_READONLY, unlike POSIX where unlinking needs only
      parent-directory write, so a readonly hardlinked file -- and its
      now-unneeded parent chain -- survives removal THERE. Clearing
      that attribute would clear it on the shared file record, i.e. on
      the user's real token: the same out-of-scope widening the skip
      exists to prevent. That Windows remnant is accepted on purpose;
      the field shape (agy writes tokens writable, and #4's actual
      conversations/*.db is nlink=1) is swept and removed normally.
      On macOS, ``setup_isolated_home`` also links the user's real
      ``~/Library/Keychains`` directory into the isolated HOME
      (``_link_macos_keychains``, process.py), so the symlink skip
      protects that keychain data as well.
    * Skipping is also sufficient: unlinking an entry needs write
      permission on its PARENT directory -- which the sweep grants via
      the walk's own directory chmods -- never on the entry's target.
    * ``os.walk`` never follows symlinked directories either
      (``followlinks=False``), so recursion cannot escape the workspace
      through a linked directory.

    Both rmtree passes ignore errors and every chmod is individually
    suppressed, so this function cannot raise.
    """
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        return
    for dirpath, dirnames, filenames in os.walk(path):
        for name in (*dirnames, *filenames):
            entry = os.path.join(dirpath, name)
            with contextlib.suppress(OSError):
                entry_stat = os.lstat(entry)
                is_link = stat.S_ISLNK(entry_stat.st_mode)
                is_dir = stat.S_ISDIR(entry_stat.st_mode)
                # The hardlink check is scoped to non-directories: an
                # ordinary directory already has st_nlink >= 2 (itself
                # plus one link per subdirectory), so skipping every
                # nlink > 1 entry would skip the whole directory tree --
                # the sweep would stop working for the read-only-tree
                # case it exists for. Only a FILE's inode can be shared
                # with the outside world (the token os.link fallback).
                if is_link or (not is_dir and entry_stat.st_nlink > 1):
                    continue
                os.chmod(
                    entry,
                    entry_stat.st_mode
                    | stat.S_IWUSR
                    | (stat.S_IXUSR if is_dir else 0),
                )
    shutil.rmtree(path, ignore_errors=True)


class AntigravityClient:
    """OpenAI-compatible client facade driving Antigravity CLI.

    Async compatibility boundary (issue #8)
    --------------------------------------
    Hermes drives this client from its async auxiliary path: with
    ``HERMES_SKIP_ASYNC_WRAP`` the client is used as-is ("already async-safe"),
    and the caller then does ``await client.chat.completions.create(...)``.
    Everything the facade returns is therefore awaitable-yielding-itself —
    ``stream._AwaitableCompletion`` for the non-streaming plan (which simply
    wraps the already-assembled response) and ``stream.AntigravityStream`` for
    the streaming plan (awaitable plus ``async for`` via ``__anext__``). Sync
    callers are untouched: the objects keep the same attributes and the same
    blocking behavior, so no existing caller needs to change.

    Known limitation (documented on purpose, not fixed here): the non-streaming
    ``create()`` executes the whole subprocess round-trip synchronously before
    it returns, so awaiting it still blocks the event loop for the request's
    duration. This is identical to the pre-existing sync behavior and to
    Hermes' own CopilotACPClient shim, and there is no plugin-side fix without
    a Hermes hook that would let ``create()`` hand back a coroutine instead of
    a finished value.
    """

    # Instruct Hermes not to wrap this client in wire transports or async adapters
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "antigravity-directsdk"
        self.base_url = base_url or AGY_MARKER_BASE_URL
        self._command = command or resolve_agy_command()
        self._args = list(args or ["--output-format", "stream-json", "--disable-slash-commands"])
        self._temp_dir = None
        if cwd:
            self._cwd = cwd
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_")
            self._cwd = self._temp_dir.name

        # Isolate agy state and session index from the user's real ~/.gemini/antigravity-cli
        self._isolated_home, self._isolated_gemini_dir = setup_isolated_home(self._cwd)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        # Valid fresh-context full-send samples only. Worker deltas and
        # concurrent oneshots are not comparable: see delivery.py and issue #29.
        self._delivery_watch = DeliveryWatch()
        self._active_processes: set[subprocess.Popen] = set()
        # RLock (NOT a plain Lock) deliberately -- issue #10.
        #
        # A suspended AntigravityStream._stream_generator forms a
        # reference cycle with the stream (stream._generator <-> generator
        # frame <-> stream), so an ABANDONED stream (consumer stopped
        # iterating without close()) is reclaimable only by the cyclic
        # GC. That pass fires at an arbitrary allocation -- possibly on
        # the SAME thread while it already holds this lock (client.close(),
        # _get_or_spawn_worker, _create_chat_completion, ...). The
        # GC-finalized generator's finally block then runs stream.close()
        # -> client._terminate_worker()/_update_worker_history() ->
        # `with self._lock` on a thread that already holds it: a plain
        # Lock self-deadlocks here (deterministic hangs were observed
        # while developing #9). RLock makes that same-thread re-entrance
        # succeed; cross-thread mutual exclusion is unchanged.
        #
        # Trade-off, documented on purpose: re-entrance can MASK a future
        # lock-ordering bug that would otherwise deadlock loudly -- a
        # code path that re-acquires _lock without expecting to already
        # hold it is now silently allowed instead of hanging. Mutating
        # shared state under _lock must therefore never call back into
        # the client (a nested _lock acquisition, a state-dependent
        # branch, another lock); the region is for flat
        # acquire -> mutate -> release only. Known, accepted exceptions
        # inside the regions below are the bounded-but-blocking process
        # calls (Popen spawn, proc.wait(timeout=2), Windows taskkill
        # without timeout): they touch no client state, and moving them
        # out of the locked region would trade the deadlock for races.
        self._lock = threading.RLock()
        self._worker_proc: subprocess.Popen | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_history: list[dict[str, Any]] = []
        self._worker_conversation_id: str | None = None
        self._conversation_history: list[dict[str, Any]] = []
        # Media rules the LIVE worker's settings.json contained at spawn:
        # agy reads permissions only at spawn, so a turn referencing new
        # media paths must respawn the worker (conversation preserved).
        self._worker_media_rules: frozenset[str] = frozenset()
        # Media files this session materialized (downloads, cache files);
        # deleted at close() so no fetched media outlives the session.
        self._materialized_media: set[str] = set()
        # Cumulative usage snapshot of the current worker session. agy 1.2.10+
        # persistent workers report cumulative session usage, so per-turn
        # deltas need this baseline. The reference is replaced on every spawn
        # and termination; the dict contents are advanced in place by the
        # owning session's stream. Each stream captures the dict of the
        # session it was created for, so a stream left over from a terminated
        # session cannot corrupt a later session's baseline.
        self._worker_usage_baseline: dict[str, int] = {}
        self._worker_lock = threading.Lock()
        self._worker_idle_timer: threading.Timer | None = None
        # Last moment each worker showed a sign of life: its spawn, or the last
        # stdout line it produced (stamped by AntigravityStream). Read by the
        # abandon check in _reap_idle_worker, which needs to tell a genuinely
        # busy worker from one whose turn died still holding _worker_lock.
        self._worker_last_activity: dict[subprocess.Popen, float] = {}
        # CPU seconds at the last sign of life: the discriminator between a
        # silent abandoned turn and a silent busy one.
        self._worker_cpu_at_activity: dict[subprocess.Popen, float] = {}

    @staticmethod
    def _resolve_real_token_path() -> Path | None:
        return resolve_real_token_path()

    def _resolve_model_and_effort(
        self,
        model: str | None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        return resolve_model_and_effort(model, reasoning_effort)

    def _child_env(self) -> dict[str, str]:
        return build_child_env(self._isolated_home)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        terminate_process(proc)

    def __enter__(self) -> "AntigravityClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        # Phase 1 -- terminate every child process BEFORE the workspace is
        # touched (issue #4): on Windows a still-running agy keeps
        # conversations/*.db open, and taskkill /F /T is asynchronous, so
        # removing the workspace first would race the OS and fail with
        # WinError 32. Ordering is therefore load-bearing, not cosmetic.
        with self._lock:
            self.is_closed = True
            self._terminate_worker_locked()
            procs = tuple(self._active_processes)
            self._active_processes.clear()
            self._worker_last_activity.clear()
            self._worker_cpu_at_activity.clear()
        for proc in procs:
            self._terminate_process(proc)
        # Phase 1.5 -- delete the media files this session materialized.
        # After every child is dead (they may still have had the files open),
        # before the workspace removal; must never raise (issue #4 contract).
        with self._lock:
            materialized = set(self._materialized_media)
            self._materialized_media.clear()
        for path in materialized:
            try:
                os.remove(path)
            except OSError:
                pass
        # Phase 2 -- remove the private workspace, retrying transient
        # failures. close() must never raise from cleanup and must never
        # leave the client half-closed (issue #4).
        self._remove_temp_dir()

    def _remove_temp_dir(self) -> None:
        """Remove the private workspace; cannot raise, cannot half-finish.

        Issue #4 (reported on Windows 11, agy 1.2.3): cleanup raised
        PermissionError [WinError 32] on conversations/*.db because the
        agy child still held the file after taskkill /F /T returned (that
        kill is asynchronous). The old code swallowed the error and gave
        up, leaking the whole workspace. Now: bounded retries with short
        backoff (see _TEMP_DIR_CLEANUP_ATTEMPTS/_BACKOFF constants above
        for the sizing), then a forced removal of whatever is unlocked.

        Live Windows verification by the reporter is still pending and
        will be requested on the issue after this merges; the retry budget
        is deliberately small so the worst case stays under ~0.5s and the
        common clean path pays nothing.

        The client is already fully closed when this runs (is_closed set,
        every process terminated in close() phase 1), and _temp_dir is
        cleared unconditionally, so a second close() is a no-op and a
        lingering workspace never blocks a fresh client on the same path.
        """
        temp_dir = self._temp_dir
        if temp_dir is None:
            return
        self._temp_dir = None
        try:
            self._cleanup_temp_dir_with_retries(temp_dir)
        except Exception as exc:
            # Defense in depth: a cleanup path must never break close().
            # Only the workspace path is logged; the isolated HOME inside
            # it holds a token link but no secret content of its own, and
            # the path itself is this client's own private directory.
            logger.debug(
                "Antigravity workspace %s cleanup raised unexpectedly: %s",
                temp_dir.name,
                exc,
            )

    def _cleanup_temp_dir_with_retries(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Try cleanup up to _TEMP_DIR_CLEANUP_ATTEMPTS times, then force.

        The loop catches EVERY exception -- not just OSError -- because the
        goal is that nothing can escape into close(): OSError
        (PermissionError/WinError 32, EBUSY, ENOTEMPTY -- the "handle
        still held" symptoms) is retried with backoff, while any other
        exception breaks out immediately and lands in the forced removal
        below. Retrying a failed TemporaryDirectory.cleanup() is safe: its
        finalizer is detached on the first call, so each attempt is a
        fresh rmtree pass.
        """
        for attempt in range(1, _TEMP_DIR_CLEANUP_ATTEMPTS + 1):
            try:
                temp_dir.cleanup()
                return
            except Exception as exc:
                retryable = isinstance(exc, OSError) and attempt < _TEMP_DIR_CLEANUP_ATTEMPTS
                if retryable:
                    logger.debug(
                        "Antigravity workspace %s cleanup attempt %d/%d failed (%s); retrying.",
                        temp_dir.name,
                        attempt,
                        _TEMP_DIR_CLEANUP_ATTEMPTS,
                        exc,
                    )
                    time.sleep(_TEMP_DIR_CLEANUP_BACKOFF_SECONDS[attempt - 1])
                    continue
                logger.debug(
                    "Antigravity workspace %s cleanup failed on attempt %d/%d (%s: %s); "
                    "forcing removal of whatever is unlocked.",
                    temp_dir.name,
                    attempt,
                    _TEMP_DIR_CLEANUP_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                )
                break
        if temp_dir.name:
            _force_rmtree(temp_dir.name)

    def _terminate_worker_locked(self, keep_conversation: bool = False) -> None:
        self._cancel_idle_timer_locked()
        if self._worker_proc is not None:
            proc = self._worker_proc
            self._worker_proc = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_history = []
            self._active_processes.discard(proc)
            self._worker_last_activity.pop(proc, None)
            self._worker_cpu_at_activity.pop(proc, None)
            self._terminate_process(proc)
        # The session is gone: any later usage from it belongs to a dead
        # session, and the next spawn must start from a clean baseline.
        self._worker_usage_baseline = {}
        # Full-send diagnostics compare only fresh-context requests, never
        # session counters or deltas. Their reference survives worker resets.
        if not keep_conversation:
            self._worker_conversation_id = None
            self._conversation_history = []

    def _terminate_worker(self, keep_conversation: bool = False) -> None:
        with self._lock:
            self._terminate_worker_locked(keep_conversation=keep_conversation)

    def _set_worker_conversation_id(self, conv_id: str) -> None:
        with self._lock:
            if conv_id:
                self._worker_conversation_id = conv_id

    def _purge_isolated_conversations(self) -> None:
        """Purge isolated conversation database files on session reset or new conversation."""
        conv_dir = self._isolated_gemini_dir / "conversations"
        if conv_dir.is_dir():
            for entry in conv_dir.glob("*.db*"):
                with contextlib.suppress(OSError):
                    if entry.is_file():
                        entry.unlink()

    def _update_worker_history(self, messages: list[dict[str, Any]] | None) -> None:
        with self._lock:
            msg_list = list(messages or [])
            self._worker_history = msg_list
            self._conversation_history = msg_list
            self._arm_idle_timer_locked()

    def _cancel_idle_timer_locked(self) -> None:
        timer = self._worker_idle_timer
        self._worker_idle_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_idle_timer_locked(self) -> None:
        """(Re)start the idle countdown; called when a worker turn finishes."""
        self._cancel_idle_timer_locked()
        proc = self._worker_proc
        idle_seconds = _worker_idle_seconds()
        if proc is None or self.is_closed or idle_seconds <= 0:
            return
        timer = threading.Timer(idle_seconds, self._reap_idle_worker, args=(proc,))
        timer.daemon = True
        self._worker_idle_timer = timer
        timer.start()

    def _reap_idle_worker(self, proc: subprocess.Popen) -> None:
        # A held worker lock means a turn is running on the worker.
        if not self._worker_lock.acquire(blocking=False):
            # Re-arm rather than give up. The countdown used to be armed only
            # by a turn that FINISHED, so a wedged or abandoned turn left the
            # worker with no timer at all and it survived the idle bound
            # entirely. Re-arming repeats the check without ever reaching the
            # terminate below, which is still what a genuinely busy worker
            # wants -- but a worker that has shown no sign of life for the
            # abandon grace is not busy, its turn is gone.
            with self._lock:
                if self._worker_proc is not proc:
                    self._worker_last_activity.pop(proc, None)
                    return
                if self._worker_is_abandoned_locked(proc):
                    last_seen = self._worker_last_activity.get(proc)
                    logger.warning(
                        "Terminating abandoned agy worker pid=%s: no output for %.0fs "
                        "while its turn still held the worker lock",
                        getattr(proc, "pid", None),
                        time.monotonic() - (last_seen if last_seen is not None else time.monotonic()),
                    )
                    self._terminate_worker_locked(keep_conversation=True)
                    return
                self._arm_idle_timer_locked()
            return
        try:
            with self._lock:
                if self._worker_proc is not proc:
                    return
                logger.info("Terminating idle agy worker pid=%s", getattr(proc, "pid", None))
                self._terminate_worker_locked(keep_conversation=True)
        finally:
            self._worker_lock.release()

    def _note_worker_activity(self, proc: subprocess.Popen) -> None:
        """Stamp a worker as alive; AntigravityStream calls this per stdout line.

        The CPU reading is taken outside the lock: it must never extend the
        critical section the reader thread shares with the timer thread.
        """
        cpu = _worker_cpu_seconds(proc)
        with self._lock:
            if self._worker_proc is proc:
                self._worker_last_activity[proc] = time.monotonic()
                if cpu is not None:
                    self._worker_cpu_at_activity[proc] = cpu

    def _worker_is_abandoned_locked(self, proc: subprocess.Popen) -> bool:
        """True when a lock-holding worker has shown no sign of life for too long.

        A held _worker_lock normally means a turn is running, and killing it
        would be a regression. But an abandoned turn never releases the lock:
        the stream generator's finally block -- the only place that releases it
        -- runs when the client/stream reference cycle is collected, and a
        lingering reference (a traceback, a cached response object) can postpone
        that indefinitely. The grace runs from the worker's last output, not
        from when the lock was taken, which nobody records -- and it is only
        acted upon when the worker has also stopped burning CPU, so a live but
        quiet turn (long reasoning, slow first token, generous read deadline) is
        never mistaken for a dead one.
        """
        abandon_after = _worker_abandon_seconds()
        if abandon_after <= 0:
            return False
        last_seen = self._worker_last_activity.get(proc)
        if last_seen is None:
            # Unknown baseline (a worker installed by an older code path): stay
            # conservative and keep re-arming rather than guess.
            return False
        if (time.monotonic() - last_seen) < abandon_after:
            return False
        # Silence is necessary but not sufficient. A live turn can be silent for
        # minutes -- a long reasoning block, a slow first token, a caller whose
        # read deadline is longer than the grace -- and it burns CPU while it
        # runs. An abandoned worker burns none. Unmeasurable CPU means unproven
        # abandonment: keep re-arming rather than risk a live turn.
        cpu_now = _worker_cpu_seconds(proc)
        cpu_then = self._worker_cpu_at_activity.get(proc)
        if cpu_now is None or cpu_then is None:
            return False
        return (cpu_now - cpu_then) < _WORKER_ABANDON_CPU_SECONDS

    def _ensure_worker_media_permissions(self, paths: set[str]) -> None:
        """Grant agy's headless read permission for this turn's media paths.

        agy auto-denies read_file/view_file without an explicit per-target
        ``read_file(<path>)`` rule in settings.json, and reads that file only
        at spawn. So: merge the rules into the isolated settings.json, and if
        a live worker was spawned without them, terminate it KEEPING the agy
        conversation — the next ``_get_or_spawn_worker`` respawns with
        ``--conversation`` and the fresh rules, so delta resume still works.
        """
        if not paths:
            return
        rules = write_media_read_permissions(self._isolated_gemini_dir, paths)
        with self._lock:
            live = (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
            )
            if live and not paths <= self._worker_media_rules:
                self._terminate_worker_locked(keep_conversation=True)
            self._worker_media_rules = frozenset(rules)

    def _get_or_spawn_worker(
        self,
        model: str,
        effort: str | None,
        conversation_id: str | None = None,
    ) -> subprocess.Popen:
        with self._lock:
            if (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
                and self._worker_model == model
                and self._worker_effort == effort
            ):
                # A continuation turn reuses the worker without respawning it.
                # Without a fresh baseline the abandon grace would be measured
                # from the PREVIOUS turn's last output, so a worker idle for
                # almost a full window and then handed a new turn could be
                # reaped mid-turn. The grace also has to restart, not merely
                # re-measure: a silent first token is normal at turn start.
                proc = self._worker_proc
                self._worker_last_activity[proc] = time.monotonic()
                cpu = _worker_cpu_seconds(proc)
                if cpu is not None:
                    self._worker_cpu_at_activity[proc] = cpu
                return proc

            self._terminate_worker_locked(keep_conversation=bool(conversation_id))

            # Multiple agy processes share one Google account; whoever
            # refreshed last wins. Re-point the isolated token at the real
            # file right before every spawn so a worker never starts from a
            # stale copy ("You are not logged into Antigravity").
            sync_isolated_token(self._isolated_home)

            cmd_args = [self._command, "--input-format", "stream-json", *self._args]
            if conversation_id:
                cmd_args.extend(["--conversation", conversation_id])
            if model:
                cmd_args.extend(["--model", model])
            if effort:
                cmd_args.extend(["--effort", effort])

            proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd,
                env=self._child_env(),
                **_own_process_group(),
            )
            start_stderr_drainer(proc)
            self._worker_proc = proc
            self._worker_model = model
            self._worker_effort = effort
            self._worker_history = list(self._conversation_history) if conversation_id else []
            # Stamp what the fresh worker's settings.json granted: agy reads
            # permissions only at spawn (verified: rewriting the file under a
            # live worker changes nothing for its next turn).
            self._worker_media_rules = read_media_rules(self._isolated_gemini_dir)
            self._worker_usage_baseline = {}
            self._active_processes.add(proc)
            self._worker_last_activity[proc] = time.monotonic()
            # Give every worker a death date at spawn, not only when a turn
            # finishes. The countdown was armed solely by
            # _update_worker_history, so a worker whose first turn never
            # completed -- an abandoned stream (consumer stopped iterating
            # without close()), a cancelled request, or a capacity retry that
            # dropped the generator -- had no timer and no close() caller, and
            # lived for the lifetime of the gateway. Hermes does not call
            # close() on an evicted client (see _DEFAULT_WORKER_IDLE_SECONDS),
            # so that was unbounded: 86 such workers were measured holding
            # 15.3 GB on a 16 GB host. A live turn still holds _worker_lock,
            # which _reap_idle_worker respects, and the turn's own completion
            # re-arms the countdown, so arming here cannot cut a turn short.
            self._arm_idle_timer_locked()
            return proc

    def _run_oneshot_completion(
        self,
        *,
        model: str,
        effort: str | None,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        stream: bool,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            media_sink=self._materialized_media,
        )
        dump_prompt_debug(
            prompt=prompt_text,
            branch="oneshot",
            model=model,
            messages=messages,
        )
        cmd_args = [self._command, *self._args]
        if model:
            cmd_args.extend(["--model", model])
        if effort:
            cmd_args.extend(["--effort", effort])

        # Same account-contention guard as the worker spawn path.
        sync_isolated_token(self._isolated_home)

        proc = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self._cwd,
            env=self._child_env(),
            **_own_process_group(),
        )
        start_stderr_drainer(proc)

        try:
            if proc.stdin:
                proc.stdin.write(prompt_text + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except OSError:
            pass

        with self._lock:
            self._active_processes.add(proc)

        stream_iter = AntigravityStream(
            proc=proc,
            client=self,
            model=model,
            timeout=timeout,
            tools=tools,
            is_worker=False,
            prompt_chars=len(prompt_text),
            # The turn's messages feed the native media-attach allowlist
            # (read_file/view_file on referenced media paths only).
            messages=messages,
            # Oneshots run concurrently when the worker is busy. They have no
            # comparable prior sample and must not mutate the worker watch.
            delivery_watch=None,
        )
        if stream:
            return stream_iter
        return collect_stream_completion(stream_iter)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        **extra_kwargs: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("AntigravityClient is closed.")

        if not is_authenticated():
            raise RuntimeError(
                "Antigravity CLI is not authenticated. Please run 'agy' in your terminal "
                "to log in with your Google account."
            )

        effort_param = reasoning_effort or extra_kwargs.get("reasoning_effort")
        resolved_model, effort = self._resolve_model_and_effort(model, effort_param)
        messages_list = list(messages or [])
        # Hermes' streaming caller passes an httpx.Timeout, not a scalar; use
        # its read budget instead of silently falling back to the default.
        if hasattr(timeout, "read"):
            read_timeout = getattr(timeout, "read")
            timeout = float("inf") if read_timeout is None else read_timeout
        effective_timeout = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout > 0
            else _DEFAULT_TIMEOUT_SECONDS
        )

        # Multimodal translation: grant agy's headless read permission for
        # every media path this turn's messages reference, BEFORE any spawn
        # decision — both the worker and the oneshot process read settings.json
        # at spawn. A live worker lacking the new paths is respawned with its
        # agy conversation preserved.
        turn_media_paths = media_reference_paths(
            messages_list, self._materialized_media
        )
        if turn_media_paths:
            self._ensure_worker_media_permissions(turn_media_paths)

        worker_acquired = self._worker_lock.acquire(blocking=False)
        if worker_acquired:
            try:
                with self._lock:
                    ref_history = (
                        self._worker_history
                        if (self._worker_proc is not None and self._worker_proc.poll() is None)
                        else self._conversation_history
                    )
                    is_continuation = _messages_match_prefix(ref_history, messages_list)
                    conv_id_to_resume = self._worker_conversation_id if is_continuation else None

                comparable_full_send = not is_continuation
                if is_continuation:
                    proc = self._get_or_spawn_worker(
                        resolved_model, effort, conversation_id=conv_id_to_resume
                    )
                    delta_msgs = messages_list[len(ref_history):]
                    prompt_payload = _format_delta_prompt(
                        delta_msgs,
                        pending_user_text=_latest_user_text(messages_list),
                        media_sink=self._materialized_media,
                    )
                    dump_prompt_debug(
                        prompt=prompt_payload,
                        branch="delta",
                        model=resolved_model,
                        messages=messages_list,
                    )
                else:
                    with self._lock:
                        if self._worker_proc is not None or self._worker_conversation_id:
                            self._terminate_worker_locked(keep_conversation=False)
                            self._purge_isolated_conversations()
                    proc = self._get_or_spawn_worker(resolved_model, effort, conversation_id=None)
                    prompt_payload = _format_messages_as_prompt(
                        messages_list,
                        model=resolved_model,
                        tools=tools,
                        tool_choice=tool_choice,
                        media_sink=self._materialized_media,
                    )
                    dump_prompt_debug(
                        prompt=prompt_payload,
                        branch="full",
                        model=resolved_model,
                        messages=messages_list,
                    )

                event_msg = {"event": "user", "message": {"content": prompt_payload}}
                try:
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    comparable_full_send = True
                    with self._lock:
                        self._terminate_worker_locked(keep_conversation=False)
                        self._purge_isolated_conversations()
                    proc = self._get_or_spawn_worker(resolved_model, effort, conversation_id=None)
                    prompt_payload = _format_messages_as_prompt(
                        messages_list,
                        model=resolved_model,
                        tools=tools,
                        tool_choice=tool_choice,
                        media_sink=self._materialized_media,
                    )
                    dump_prompt_debug(
                        prompt=prompt_payload,
                        branch="retry",
                        model=resolved_model,
                        messages=messages_list,
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()

                with self._lock:
                    worker_usage_baseline = self._worker_usage_baseline
                stream_iter = AntigravityStream(
                    proc=proc,
                    client=self,
                    model=resolved_model,
                    timeout=effective_timeout,
                    tools=tools,
                    is_worker=True,
                    worker_lock=self._worker_lock,
                    worker_lock_held=worker_acquired,
                    messages=messages_list,
                    usage_baseline=worker_usage_baseline,
                    prompt_chars=len(prompt_payload),
                    delivery_watch=self._delivery_watch if comparable_full_send else None,
                    delivery_sample_key=f"{resolved_model}:{effort}",
                )
                if stream:
                    return stream_iter
                completion: Any = None
                auth_retried = False
                while True:
                    completion = collect_stream_completion(stream_iter)
                    if auth_retried or not _empty_no_result(completion):
                        break
                    if not _stderr_shows_auth_contention(proc):
                        break
                    # Token contention with another agy session on this
                    # account: the worker produced no result and its stderr
                    # carries the token-source failure. Re-sync the token and
                    # retry the turn ONCE with a fresh worker, keeping the
                    # agy conversation so delta resume still works.
                    logger.warning(
                        "Antigravity worker ended with no result amid token-source auth errors; "
                        "re-syncing token and retrying once."
                    )
                    auth_retried = True
                    sync_isolated_token(self._isolated_home)
                    with self._lock:
                        self._terminate_worker_locked(keep_conversation=True)
                    proc = self._get_or_spawn_worker(
                        resolved_model, effort, conversation_id=self._worker_conversation_id
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    try:
                        if proc.stdin:
                            proc.stdin.write(json.dumps(event_msg) + "\n")
                            proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break
                    with self._lock:
                        worker_usage_baseline = self._worker_usage_baseline
                    stream_iter = AntigravityStream(
                        proc=proc,
                        client=self,
                        model=resolved_model,
                        timeout=effective_timeout,
                        tools=tools,
                        is_worker=True,
                        worker_lock=self._worker_lock,
                        worker_lock_held=True,
                        messages=messages_list,
                        usage_baseline=worker_usage_baseline,
                        prompt_chars=len(prompt_payload),
                        delivery_watch=self._delivery_watch if comparable_full_send else None,
                        delivery_sample_key=f"{resolved_model}:{effort}",
                    )
                try:
                    return completion
                finally:
                    if self._worker_lock.locked():
                        self._worker_lock.release()
            except Exception:
                self._terminate_worker()
                if self._worker_lock.locked():
                    self._worker_lock.release()
                raise
        else:
            return self._run_oneshot_completion(
                model=resolved_model,
                effort=effort,
                messages=messages_list,
                timeout=effective_timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
            )
