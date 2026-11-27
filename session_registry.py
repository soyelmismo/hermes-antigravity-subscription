"""Process-wide store of agy conversation snapshots, keyed by Hermes session.

Why this exists
---------------
``AntigravityClient`` keeps the agy conversation, the history already sent and
the conversation id on the client instance. Hermes' gateway builds a new agent
(and a new request-scoped client) for every incoming message, and runs its
background review on yet another one, so every one of those starts with an
empty history and re-sends the whole prompt as a split full send (6 parts and
~350k characters per group message measured with Gemini routes).

The conversation itself is one SQLite file in the isolated HOME
(``conversations/<id>.db``). This module keeps point-in-time copies of that
file in a private directory together with the message history it contains, so
a later client can restore the file into its own HOME, respawn agy with
``--conversation <id>`` and send only the new messages.

Design rules
------------
* Snapshots are immutable. A client that adopts one copies the file out and
  publishes its own, newer snapshot after the turn; the parent stays. That is
  what keeps forked requests (Hermes' background review shares the session id
  and the prefix, then appends its own instructions) from polluting the main
  line: the next main message matches the untouched parent again. It also means
  two clients can adopt the same snapshot concurrently without any lock.
* The key is ``sha256(hermes session id, model, effort, account token path,
  system prompt)``. A model, effort or account change therefore never resumes
  an unrelated conversation, and an auxiliary call that shares a session id but
  not the system prompt stays on its own line.
* A snapshot is adopted only when the incoming messages strictly extend its
  history (``prompt._messages_match_prefix``): anything else is a full send,
  exactly as before.
* Snapshots expire after ``ANTIGRAVITY_SESSION_TTL_SECONDS`` (default 6 h), at
  most three are kept per key and sixteen in total (least recently used go
  first). The directory is removed at interpreter exit, and directories of dead
  processes are swept the next time a registry creates its own.
* ``ANTIGRAVITY_SESSION_RESUME=0`` turns the feature off.

Nothing here survives a restart of the Hermes process: the metadata lives in
memory on purpose.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .prompt import _messages_match_prefix
except ImportError:
    from prompt import _messages_match_prefix

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 6 * 3600.0
_MAX_PER_KEY = 3
_MAX_TOTAL = 16
_STORE_PREFIX = "hermes_agy_sessions_"
_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def session_resume_enabled() -> bool:
    raw = os.environ.get("ANTIGRAVITY_SESSION_RESUME", "").strip().lower()
    return raw not in _OFF_VALUES and session_ttl_seconds() > 0


def session_ttl_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_SESSION_TTL_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_TTL_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TTL_SECONDS


def session_key(
    *,
    session_id: str | None,
    model: str | None,
    effort: str | None,
    account: str | None,
    messages: Sequence[dict[str, Any]],
) -> str | None:
    """Stable key of one agy conversation line, or None when it has no owner."""
    if not session_id:
        return None
    system = ""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system = json.dumps(messages[0].get("content"), sort_keys=True, default=str)
    parts = (
        str(session_id),
        model or "",
        effort or "",
        account or "",
        hashlib.sha256(system.encode("utf-8", errors="replace")).hexdigest(),
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()


def slim_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep copy of the only fields prefix matching reads.

    The registry outlives the agent that produced the messages, and Hermes may
    keep mutating its own dicts, so the stored history must not alias them.
    """
    slim: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            slim.append(msg)
            continue
        slim.append(
            {
                "role": msg.get("role"),
                "content": copy.deepcopy(msg.get("content")),
                "tool_calls": copy.deepcopy(msg.get("tool_calls")),
            }
        )
    return slim


@dataclass
class SessionSnapshot:
    snapshot_id: str
    key: str
    conversation_id: str
    history: list[dict[str, Any]]
    last_assistant: dict[str, Any] | None
    usage_baseline: dict[str, int]
    db_path: Path
    last_used: float


def _backup_sqlite(source: Path, dest: Path) -> None:
    """Consistent copy of a (possibly live, WAL-mode) SQLite file."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=5.0)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


class SessionRegistry:
    def __init__(
        self,
        *,
        store_dir: Path | str | None = None,
        max_per_key: int = _MAX_PER_KEY,
        max_total: int = _MAX_TOTAL,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str, SessionSnapshot] = {}
        self._fixed_dir = Path(store_dir) if store_dir else None
        self._dir: Path | None = None
        self._max_per_key = max(1, max_per_key)
        self._max_total = max(1, max_total)
        self._atexit_registered = False

    # -- storage -----------------------------------------------------------

    def _store(self) -> Path:
        """The private directory, recreated if something removed it."""
        with self._lock:
            if self._dir is None:
                if self._fixed_dir is not None:
                    self._fixed_dir.mkdir(parents=True, exist_ok=True)
                    self._dir = self._fixed_dir
                else:
                    self._sweep_dead_stores()
                    self._dir = Path(
                        tempfile.mkdtemp(prefix=f"{_STORE_PREFIX}{os.getpid()}_")
                    )
                if not self._atexit_registered:
                    atexit.register(self.shutdown)
                    self._atexit_registered = True
            self._dir.mkdir(parents=True, exist_ok=True)
            return self._dir

    @staticmethod
    def _sweep_dead_stores() -> None:
        """Remove store directories left behind by processes that no longer exist."""
        root = Path(tempfile.gettempdir())
        with contextlib.suppress(OSError):
            for entry in root.glob(f"{_STORE_PREFIX}*"):
                pid_text = entry.name[len(_STORE_PREFIX):].split("_", 1)[0]
                if not pid_text.isdigit() or int(pid_text) == os.getpid():
                    continue
                try:
                    os.kill(int(pid_text), 0)
                except ProcessLookupError:
                    shutil.rmtree(entry, ignore_errors=True)
                except OSError:
                    continue  # exists but is not ours to signal: leave it

    def shutdown(self) -> None:
        with self._lock:
            self._snapshots.clear()
            directory, self._dir = self._dir, None
        if directory is not None and self._fixed_dir is None:
            shutil.rmtree(directory, ignore_errors=True)

    def clear(self) -> None:
        with self._lock:
            doomed = list(self._snapshots.values())
            self._snapshots.clear()
        for snap in doomed:
            self._unlink(snap.db_path)

    @staticmethod
    def _unlink(path: Path) -> None:
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                Path(f"{path}{suffix}").unlink()

    # -- index -------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def _expire_locked(self, now: float) -> list[SessionSnapshot]:
        ttl = session_ttl_seconds()
        doomed = [s for s in self._snapshots.values() if now - s.last_used > ttl]
        for snap in doomed:
            self._snapshots.pop(snap.snapshot_id, None)
        return doomed

    def _enforce_caps_locked(self) -> list[SessionSnapshot]:
        doomed: list[SessionSnapshot] = []
        by_key: dict[str, list[SessionSnapshot]] = {}
        for snap in self._snapshots.values():
            by_key.setdefault(snap.key, []).append(snap)
        for snaps in by_key.values():
            snaps.sort(key=lambda s: s.last_used, reverse=True)
            doomed.extend(snaps[self._max_per_key:])
        for snap in doomed:
            self._snapshots.pop(snap.snapshot_id, None)
        overflow = len(self._snapshots) - self._max_total
        if overflow > 0:
            oldest = sorted(self._snapshots.values(), key=lambda s: s.last_used)[:overflow]
            for snap in oldest:
                self._snapshots.pop(snap.snapshot_id, None)
            doomed.extend(oldest)
        return doomed

    def lookup(self, key: str, messages: Sequence[dict[str, Any]]) -> SessionSnapshot | None:
        """Longest snapshot of ``key`` whose history ``messages`` strictly extend."""
        now = time.monotonic()
        with self._lock:
            doomed = self._expire_locked(now)
            best: SessionSnapshot | None = None
            for snap in self._snapshots.values():
                if snap.key != key:
                    continue
                if best is not None and len(snap.history) <= len(best.history):
                    continue
                if _messages_match_prefix(snap.history, messages):
                    best = snap
            if best is not None:
                best.last_used = now
        for snap in doomed:
            self._unlink(snap.db_path)
        return best

    def restore(self, snap: SessionSnapshot, conversations_dir: Path) -> bool:
        """Copy the snapshot's conversation file into ``conversations_dir``.

        False when the snapshot's file is gone (evicted meanwhile): the caller
        then falls back to a full send.
        """
        dest = Path(conversations_dir) / f"{snap.conversation_id}.db"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", "-wal", "-shm"):
                with contextlib.suppress(OSError):
                    Path(f"{dest}{suffix}").unlink()
            shutil.copyfile(snap.db_path, dest)
            os.chmod(dest, 0o600)
        except OSError:
            logger.debug("Session snapshot %s could not be restored", snap.snapshot_id, exc_info=True)
            with self._lock:
                self._snapshots.pop(snap.snapshot_id, None)
            return False
        return True

    def publish(
        self,
        key: str,
        *,
        conversation_id: str,
        history: Sequence[dict[str, Any]],
        last_assistant: dict[str, Any] | None,
        usage_baseline: dict[str, int] | None,
        source_db: Path,
        replace: str | None = None,
    ) -> str | None:
        """Snapshot ``source_db`` with the history it holds; returns the snapshot id.

        ``replace`` names a snapshot this caller published earlier for the very
        same state (a refresh after the worker exited): it keeps its id and its
        file is swapped in place. Never a snapshot taken from another client:
        those stay immutable.
        """
        source_db = Path(source_db)
        if not conversation_id or not source_db.is_file():
            return None
        store = self._store()
        with self._lock:
            existing = self._snapshots.get(replace) if replace else None
        # A refresh keeps its id and path (the file is swapped atomically, so an
        # adopter mid-copy keeps the old inode); anything else is a new snapshot.
        snapshot_id = existing.snapshot_id if existing is not None else uuid.uuid4().hex
        target = existing.db_path if existing is not None else store / f"{snapshot_id}.db"
        staging = store / f"{snapshot_id}.{uuid.uuid4().hex}.tmp"
        try:
            _backup_sqlite(source_db, staging)
            os.replace(staging, target)
        except (sqlite3.Error, OSError):
            logger.debug("Session snapshot of %s failed", source_db, exc_info=True)
            self._unlink(staging)
            return None
        now = time.monotonic()
        snap = SessionSnapshot(
            snapshot_id=snapshot_id,
            key=key,
            conversation_id=conversation_id,
            history=slim_messages(history),
            last_assistant=copy.deepcopy(last_assistant),
            usage_baseline=dict(usage_baseline or {}),
            db_path=target,
            last_used=now,
        )
        with self._lock:
            self._snapshots[snapshot_id] = snap
            doomed = self._expire_locked(now) + self._enforce_caps_locked()
        for old in doomed:
            if old.snapshot_id != snapshot_id:
                self._unlink(old.db_path)
        return snapshot_id


_REGISTRY = SessionRegistry()


def get_registry() -> SessionRegistry:
    return _REGISTRY
