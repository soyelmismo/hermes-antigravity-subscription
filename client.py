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
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        terminate_process,
    )
    from .prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
    )
    from .stream import AntigravityStream, collect_stream_completion
    from .accounts import (
        clear_session_pin,
        fetch_usage_for_home,
        get_rotation_mode,
        is_quota_error,
        acquire_lease,
        lease_account,
        release_lease,
        parse_reset_time,
        is_failover_enabled,
        has_registered_accounts,
        pick_account,
        set_cooldown,
        record_serving_account,
        update_last_used,
    )
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
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        terminate_process,
    )
    from prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
    )
    from stream import AntigravityStream, collect_stream_completion
    from accounts import (
        clear_session_pin,
        fetch_usage_for_home,
        get_rotation_mode,
        is_quota_error,
        acquire_lease,
        lease_account,
        release_lease,
        parse_reset_time,
        is_failover_enabled,
        has_registered_accounts,
        pick_account,
        set_cooldown,
        record_serving_account,
        update_last_used,
    )

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300.0

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


def _resolve_cooldown_reset(account: dict[str, Any], model: str | None) -> float:
    home_dir = account.get("home_dir", "")
    m_lower = (model or "").lower()
    group_key = "claude_gpt" if ("claude" in m_lower or "gpt" in m_lower) else "gemini"
    usage = fetch_usage_for_home(home_dir, cached=False)
    if usage and group_key in usage:
        reset_str = usage[group_key].get("5h", {}).get("reset_time")
        reset_ts = parse_reset_time(reset_str)
        if reset_ts and reset_ts > time.time():
            return reset_ts
    return time.time() + 900.0


def _format_account_usage(account: dict[str, Any] | None, model: str | None) -> str:
    if not account:
        return "none (0%)"
    label = account.get("label", "unknown")
    home_dir = account.get("home_dir", "")
    m_lower = (model or "").lower()
    group_key = "claude_gpt" if ("claude" in m_lower or "gpt" in m_lower) else "gemini"
    group_label = "claude" if group_key == "claude_gpt" else "gemini"
    try:
        usage = fetch_usage_for_home(home_dir, cached=True)
    except Exception:
        usage = None
    pct = 0
    if usage and group_key in usage:
        f_5h = usage[group_key].get("5h", {}).get("remaining_fraction")
        if f_5h is not None:
            pct = int(f_5h * 100)
    return f"{label} ({group_label} 5h {pct}%)"


def _log_rotation_failover(
    prev_account: dict[str, Any],
    next_account: dict[str, Any] | None,
    model: str | None,
) -> None:
    prev_fmt = _format_account_usage(prev_account, model)
    prev_part = prev_fmt.replace(" (", " exhausted (", 1)
    next_part = _format_account_usage(next_account, model)
    logger.warning("[agy-rotate] %s -> %s", prev_part, next_part)


def resolve_ambient_session_id(explicit: str | None = None) -> str | None:
    """Resolve the session id binding this client's calls to one rotation pin.

    Precedence: explicit arg (host passes ``session_id=`` on create() when it
    knows it), then the ambient Hermes runtime (auxiliary_client publishes the
    live main turn's ``session_id`` and rotation-stable ``cache_scope`` via
    ``set_runtime_main``; portal_tags publishes the conversation root), then the
    caller's ``session_id`` create() kwarg. All lookups are best-effort —
    the host may be absent entirely (tests, standalone use): return None and
    rotation simply runs unpinned.
    """
    sid = str(explicit or "").strip()
    if sid:
        return sid
    try:
        from agent.auxiliary_client import _runtime_main_value

        sid = str(_runtime_main_value("cache_scope") or _runtime_main_value("session_id") or "").strip()
        if sid:
            return sid
    except Exception:
        pass
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context

        sid = str(get_affinity_scope() or get_conversation_context() or "").strip()
        if sid:
            return sid
    except Exception:
        pass
    return None


class _RotatingStreamWrapper:
    """Stream wrapper with turn 1 failover for quota exhaustion across accounts."""

    def __init__(
        self,
        client: Any,
        resolved_model: str,
        effort: str | None,
        messages_list: list[dict[str, Any]],
        effective_timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        initial_account: dict[str, Any],
        initial_stream: Any,
        tried_labels: set[str],
        max_attempts: int = 3,
        session_id: str | None = None,
        force_mode: str | None = None,
        lease_label: str | None = None,
    ) -> None:
        self.client = client
        self.resolved_model = resolved_model
        self.effort = effort
        self.messages_list = messages_list
        self.effective_timeout = effective_timeout
        self.tools = tools
        self.tool_choice = tool_choice
        self.current_account = initial_account
        self.current_stream = iter(initial_stream)
        self.tried_labels = set(tried_labels)
        self.max_attempts = max_attempts
        self.session_id = session_id
        self.force_mode = force_mode
        # The streamed turn owns its account lease: it stays held until the
        # stream is exhausted or closed, not just until the wrapper is built.
        self._lease_label = (lease_label or "").strip() or None
        self._lease_held = self._lease_label is not None
        self.chunks_yielded = 0

    def _failover(self, exc: Exception) -> bool:
        if self.chunks_yielded > 0 or not is_quota_error(exc) or len(self.tried_labels) >= self.max_attempts:
            return False

        with contextlib.suppress(Exception):
            self.current_stream.close()

        # The host default account has no registry entry: nothing to cool down,
        # and no label to exclude.
        current_label = str(self.current_account.get("label") or "").strip()
        if current_label:
            cooldown_ts = _resolve_cooldown_reset(self.current_account, self.resolved_model)
            set_cooldown(current_label, until=cooldown_ts)

        next_acc = pick_account(
            model=self.resolved_model,
            exclude_labels=self.tried_labels,
            session_id=self.session_id,
            force_mode=self.force_mode,
        )
        _log_rotation_failover(self.current_account, next_acc, self.resolved_model)
        if not next_acc:
            return False

        self.tried_labels.add(next_acc["label"])
        # The turn moved to another account, so the lease must move with it or
        # the account that failed over stays leased forever.
        self._move_lease(str(next_acc.get("label") or ""))
        self.current_account = next_acc
        self.current_stream = iter(
            self.client._execute_chat_completion(
                resolved_model=self.resolved_model,
                effort=self.effort,
                messages_list=self.messages_list,
                effective_timeout=self.effective_timeout,
                tools=self.tools,
                tool_choice=self.tool_choice,
                stream=True,
                home_dir=next_acc["home_dir"],
                account_label=next_acc["label"],
            )
        )
        return True

    def _release_lease(self) -> None:
        """Release the held account lease once; later calls are no-ops."""
        if self._lease_held and self._lease_label:
            release_lease(self._lease_label)
        self._lease_held = False

    def _move_lease(self, label: str) -> None:
        """Move the lease to the account this stream just failed over to."""
        self._release_lease()
        norm = (label or "").strip()
        if norm:
            acquire_lease(norm)
            self._lease_label = norm
            self._lease_held = True

    def __del__(self) -> None:
        # Last resort for a consumer that abandons the stream without closing it.
        with contextlib.suppress(Exception):
            self._release_lease()

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        while True:
            try:
                chunk = next(self.current_stream)
                self.chunks_yielded += 1
                return chunk
            except StopIteration:
                self._release_lease()
                raise
            except Exception as exc:
                if self._failover(exc):
                    continue
                self._release_lease()
                raise

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        while True:
            try:
                chunk = await self.current_stream.__anext__()
                self.chunks_yielded += 1
                return chunk
            except StopAsyncIteration:
                self._release_lease()
                raise
            except Exception as exc:
                if self._failover(exc):
                    continue
                self._release_lease()
                raise

    def __await__(self) -> Any:
        try:
            from .stream import _ready
        except ImportError:
            from stream import _ready
        return _ready(self).__await__()

    def close(self) -> None:
        """Close the underlying stream and drop the held account lease."""
        with contextlib.suppress(Exception):
            self.current_stream.close()
        self._release_lease()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current_stream, name)


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
        self._worker_home_dir: Path | str | None = None
        self._worker_account_label: str | None = None
        self._current_account_gemini_dir: Path | None = None
        # Cumulative usage snapshot of the current worker session. agy 1.2.10+
        # persistent workers report cumulative session usage, so per-turn
        # deltas need this baseline. The reference is replaced on every spawn
        # and termination; the dict contents are advanced in place by the
        # owning session's stream. Each stream captures the dict of the
        # session it was created for, so a stream left over from a terminated
        # session cannot corrupt a later session's baseline.
        self._worker_usage_baseline: dict[str, int] = {}
        self._worker_lock = threading.Lock()

    @staticmethod
    def _resolve_real_token_path() -> Path | None:
        return resolve_real_token_path()

    def _resolve_model_and_effort(
        self,
        model: str | None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        return resolve_model_and_effort(model, reasoning_effort)

    def _child_env(self, home_dir: Path | str | None = None) -> dict[str, str]:
        target_home = home_dir if home_dir is not None else self._isolated_home
        return build_child_env(target_home)

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
        for proc in procs:
            self._terminate_process(proc)
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

    def _terminate_worker_locked(self) -> None:
        if self._worker_proc is not None:
            proc = self._worker_proc
            self._worker_proc = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_history = []
            self._worker_home_dir = None
            self._worker_account_label = None
            self._active_processes.discard(proc)
            self._terminate_process(proc)
        # The session is gone: any later usage from it belongs to a dead
        # session, and the next spawn must start from a clean baseline.
        self._worker_usage_baseline = {}

    def _terminate_worker(self) -> None:
        with self._lock:
            self._terminate_worker_locked()

    def _update_worker_history(self, messages: list[dict[str, Any]] | None) -> None:
        with self._lock:
            self._worker_history = list(messages or [])

    def _get_or_spawn_worker(
        self,
        model: str,
        effort: str | None,
        home_dir: Path | str | None = None,
        account_label: str | None = None,
    ) -> subprocess.Popen:
        with self._lock:
            if (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
                and self._worker_model == model
                and self._worker_effort == effort
                and self._worker_home_dir == home_dir
                and self._worker_account_label == account_label
            ):
                return self._worker_proc

            self._terminate_worker_locked()

            cmd_args = [self._command, "--input-format", "stream-json", *self._args]
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
                env=self._child_env(home_dir),
                **_own_process_group(),
            )
            self._worker_proc = proc
            self._worker_model = model
            self._worker_effort = effort
            self._worker_home_dir = home_dir
            self._worker_account_label = account_label
            self._worker_history = []
            self._worker_usage_baseline = {}
            self._active_processes.add(proc)
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
        home_dir: Path | str | None = None,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages, model=model, tools=tools, tool_choice=tool_choice
        )
        cmd_args = [self._command, *self._args]
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
            env=self._child_env(home_dir),
            **_own_process_group(),
        )

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
        )
        if stream:
            return stream_iter
        return collect_stream_completion(stream_iter)

    def _execute_chat_completion(
        self,
        *,
        resolved_model: str,
        effort: str | None,
        messages_list: list[dict[str, Any]],
        effective_timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        stream: bool,
        home_dir: Path | str | None = None,
        account_label: str | None = None,
    ) -> Any:
        if home_dir:
            self._current_account_gemini_dir = Path(home_dir) / ".gemini" / "antigravity-cli"
        else:
            self._current_account_gemini_dir = None
        # Every route (rotation, host default, quota failover) runs through here, so this
        # is the one place that knows which account actually serves the request.
        record_serving_account(account_label, model=resolved_model)

        worker_acquired = self._worker_lock.acquire(blocking=False)
        if worker_acquired:
            try:
                proc = self._get_or_spawn_worker(
                    resolved_model, effort, home_dir=home_dir, account_label=account_label
                )
                with self._lock:
                    is_continuation = _messages_match_prefix(self._worker_history, messages_list)

                if is_continuation:
                    delta_msgs = messages_list[len(self._worker_history):]
                    prompt_payload = _format_delta_prompt(delta_msgs)
                else:
                    if self._worker_history:
                        self._terminate_worker()
                        proc = self._get_or_spawn_worker(
                            resolved_model, effort, home_dir=home_dir, account_label=account_label
                        )
                    prompt_payload = _format_messages_as_prompt(
                        messages_list, model=resolved_model, tools=tools, tool_choice=tool_choice
                    )

                event_msg = {"event": "user", "message": {"content": prompt_payload}}
                try:
                    if proc.stdin:
                        proc.stdin.write(json.dumps(event_msg) + "\n")
                        proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._terminate_worker()
                    proc = self._get_or_spawn_worker(
                        resolved_model, effort, home_dir=home_dir, account_label=account_label
                    )
                    prompt_payload = _format_messages_as_prompt(
                        messages_list, model=resolved_model, tools=tools, tool_choice=tool_choice
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    if proc.stdin:
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
                )
                if stream:
                    return stream_iter
                try:
                    return collect_stream_completion(stream_iter)
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
                home_dir=home_dir,
            )

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
        session_id: str | None = None,
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
        effective_timeout = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout > 0
            else _DEFAULT_TIMEOUT_SECONDS
        )
        # Session stickiness binding: explicit kwarg, else the ambient Hermes
        # runtime (aux runtime mirrors / conversation context). Purely advisory
        # for rotation; a None result just means an unpinned pick.
        eff_session_id = resolve_ambient_session_id(
            session_id or extra_kwargs.get("session_id")
        )

        rotation_mode = get_rotation_mode()
        max_attempts = 3
        if rotation_mode == "off":
            # Rotation is off (host default account). A quota error must still move
            # to a healthy registered account instead of letting Hermes degrade to
            # another model — but only when accounts exist to fail over to.
            if not is_failover_enabled() or not has_registered_accounts():
                return self._execute_chat_completion(
                    resolved_model=resolved_model,
                    effort=effort,
                    messages_list=messages_list,
                    effective_timeout=effective_timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=stream,
                )
            host_account: dict[str, Any] = {
                "label": "",
                "home_dir": "",
                "enabled": True,
                "eligible": True,
            }
            if stream:
                raw_host_stream = self._execute_chat_completion(
                    resolved_model=resolved_model,
                    effort=effort,
                    messages_list=messages_list,
                    effective_timeout=effective_timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                )
                return _RotatingStreamWrapper(
                    client=self,
                    resolved_model=resolved_model,
                    effort=effort,
                    messages_list=messages_list,
                    effective_timeout=effective_timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                    initial_account=host_account,
                    initial_stream=raw_host_stream,
                    tried_labels={""},
                    max_attempts=max_attempts,
                    session_id=eff_session_id,
                    force_mode="quota",
                )
            try:
                return self._execute_chat_completion(
                    resolved_model=resolved_model,
                    effort=effort,
                    messages_list=messages_list,
                    effective_timeout=effective_timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=False,
                )
            except Exception as exc:
                if not is_quota_error(exc):
                    raise
                next_acc = pick_account(
                    model=resolved_model,
                    exclude_labels={""},
                    session_id=eff_session_id,
                    force_mode="quota",
                )
                _log_rotation_failover(host_account, next_acc, resolved_model)
                if not next_acc:
                    raise
                update_last_used(next_acc["label"])
                return self._execute_chat_completion(
                    resolved_model=resolved_model,
                    effort=effort,
                    messages_list=messages_list,
                    effective_timeout=effective_timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=False,
                    home_dir=next_acc["home_dir"],
                    account_label=next_acc["label"],
                )
        tried_labels: set[str] = set()
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            account = pick_account(
                model=resolved_model,
                exclude_labels=tried_labels,
                session_id=eff_session_id,
            )
            if not account:
                if not tried_labels:
                    # Fail-open: no accounts configured or registry error
                    return self._execute_chat_completion(
                        resolved_model=resolved_model,
                        effort=effort,
                        messages_list=messages_list,
                        effective_timeout=effective_timeout,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream=stream,
                    )
                if last_exc:
                    raise last_exc
                raise RuntimeError("All eligible Antigravity accounts exhausted.")

            label = account["label"]
            home_dir = account["home_dir"]
            tried_labels.add(label)

            if stream:
                # Acquire before the call so two concurrent turns cannot pick
                # the same account; ownership transfers to the returned stream.
                acquire_lease(label)
                try:
                    raw_stream = self._execute_chat_completion(
                        resolved_model=resolved_model,
                        effort=effort,
                        messages_list=messages_list,
                        effective_timeout=effective_timeout,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream=True,
                        home_dir=home_dir,
                        account_label=label,
                    )
                    wrapper = _RotatingStreamWrapper(
                        client=self,
                        resolved_model=resolved_model,
                        effort=effort,
                        messages_list=messages_list,
                        effective_timeout=effective_timeout,
                        tools=tools,
                        tool_choice=tool_choice,
                        initial_account=account,
                        initial_stream=raw_stream,
                        tried_labels=tried_labels,
                        max_attempts=max_attempts,
                        session_id=eff_session_id,
                        lease_label=label,
                    )
                    update_last_used(label)
                    return wrapper
                except BaseException:
                    # The stream never came back: nothing owns the lease yet.
                    release_lease(label)
                    raise

            try:
                with lease_account(label):
                    res = self._execute_chat_completion(
                        resolved_model=resolved_model,
                        effort=effort,
                        messages_list=messages_list,
                        effective_timeout=effective_timeout,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream=False,
                        home_dir=home_dir,
                        account_label=label,
                    )
                    update_last_used(label)
                    return res
            except Exception as exc:
                if is_quota_error(exc):
                    last_exc = exc
                    cooldown_ts = _resolve_cooldown_reset(account, resolved_model)
                    set_cooldown(label, until=cooldown_ts)
                    next_acc = pick_account(
                        model=resolved_model,
                        exclude_labels=tried_labels,
                        session_id=eff_session_id,
                    )
                    _log_rotation_failover(account, next_acc, resolved_model)
                    continue
                raise

        if last_exc:
            raise last_exc
