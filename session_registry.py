"""Persistent store of agy conversation snapshots, keyed by Hermes session.

Why this exists
---------------
``AntigravityClient`` keeps the agy conversation, the history already sent and
the conversation id on the client instance. Hermes' gateway builds a new agent
(and a new request-scoped client) for every incoming message, and runs its
background review on yet another one, so every one of those starts with an
empty history and re-sends the whole prompt as a split full send (6 parts and
~350k characters per group message measured with Gemini routes). A restart of
the Hermes process (gateway, dashboard, cron, a second ``hermes chat
--resume``) loses even the in-process state.

The conversation itself is one SQLite file in the isolated HOME
(``conversations/<id>.db``). This module keeps point-in-time copies of that
file on disk together with the message history it contains, so a later client,
in this process or in another one, can restore the file into its own HOME,
respawn agy with ``--conversation <id>`` and send only the new messages.

Storage
-------
``$ANTIGRAVITY_SESSION_STORE_DIR``, else ``$HERMES_HOME/cache/agy-session-
snapshots`` (``~/.hermes`` when ``HERMES_HOME`` is unset). The directory is
0700 and every file in it 0600: the files hold whole conversations.

* ``<snapshot>.db``: the conversation file (SQLite online backup).
* ``<snapshot>.json``: history, last assistant turn and usage baseline. Written
  once (temp file + rename) and never modified, except by the publisher's own
  refresh at close(), which renames a new file over it.
* ``index.json``: one small row per snapshot (key, conversation id, history
  length, last use, bytes). Rewritten with temp file + rename under an
  exclusive ``fcntl.flock`` on ``index.lock`` (a separate file, because rename
  replaces the index inode).

JSON rather than SQLite for the index: it stays tiny (one row per snapshot,
histories live in their own write-once files), so a full rewrite per change
costs microseconds; temp file + rename can never leave a half-written index;
a corrupt or foreign file is simply treated as empty (worst case: full sends
until the next publish), with no journal, schema or WAL recovery to reason
about across four processes; and it is readable with ``cat`` when debugging.

Design rules
------------
* Snapshots are immutable. A client that adopts one copies the file out and
  publishes its own, newer snapshot after the turn; the parent stays. That is
  what keeps forked requests (Hermes' background review shares the session id
  and the prefix, then appends its own instructions) from polluting the main
  line: the next main message matches the untouched parent again. It also means
  two clients can adopt the same snapshot concurrently.
* The key is ``sha256(hermes session id, model, effort, account token path,
  system prompt)``. A model, effort or account change therefore never resumes
  an unrelated conversation, and an auxiliary call that shares a session id but
  not the system prompt stays on its own line.
* One live user per conversation id, across processes. While a client has a
  turn in flight on conversation X (adopted or its own), nobody else may adopt
  a snapshot of X (agy keeps only one of two writers on one id); such a request
  is a full send. The claim is an exclusive ``flock`` on ``claims/<id>.lock``
  held by an open descriptor, with the holder's pid and time written inside.
  It is released when the turn has been snapshotted, when the worker is
  terminated or reaped, at close(), when the owning client is garbage
  collected, and by the kernel when the holding process dies. A claim older
  than ``_CLAIM_MAX_SECONDS`` is ignored, so no failure path can hold an id
  forever.
* A snapshot is adopted only when the incoming messages strictly extend its
  history (``prompt._messages_match_prefix``): anything else is a full send.
* Clean-up runs when a process first opens the store and after every publish:
  snapshots idle longer than ``ANTIGRAVITY_SESSION_TTL_SECONDS`` (default 6 h)
  go first, then the least recently used beyond three per key,
  ``ANTIGRAVITY_SESSION_MAX_SNAPSHOTS`` (default 64) in total and
  ``ANTIGRAVITY_SESSION_MAX_BYTES`` (default 512 MiB) on disk. Files without an
  index row (a crashed publish) and the per-process temp directories of the
  previous in-memory design are removed too.
* ``ANTIGRAVITY_SESSION_RESUME=0`` turns the feature off.
* Every failure is contained: an unreadable index or snapshot behaves as "no
  snapshot", which is the full send the turn would have been anyway.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    import fcntl
except ImportError:  # Windows: claims and index updates stay process-local.
    fcntl = None  # type: ignore[assignment]

try:
    from .prompt import _messages_match_prefix
except ImportError:
    from prompt import _messages_match_prefix

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 6 * 3600.0
_MAX_PER_KEY = 3
_MAX_TOTAL = 64
_MAX_BYTES = 512 * 1024 * 1024
# Prefix of the per-process temp directories of the in-memory design.
_LEGACY_STORE_PREFIX = "hermes_agy_sessions_"
_STORE_DIRNAME = "agy-session-snapshots"
_INDEX_NAME = "index.json"
_INDEX_LOCK_NAME = "index.lock"
_CLAIMS_DIRNAME = "claims"
_INDEX_VERSION = 1
# A publish writes its .db and .json before the index row: files younger than
# this without a row may belong to a publish in flight in another process.
_ORPHAN_GRACE_SECONDS = 600.0
# Longest a claim is honoured: far beyond any single turn, so it only matters
# when a release was skipped by a failure nobody anticipated.
_CLAIM_MAX_SECONDS = 3600.0
_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def session_resume_enabled() -> bool:
    raw = os.environ.get("ANTIGRAVITY_SESSION_RESUME", "").strip().lower()
    return raw not in _OFF_VALUES and session_ttl_seconds() > 0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def session_ttl_seconds() -> float:
    return _env_float("ANTIGRAVITY_SESSION_TTL_SECONDS", _DEFAULT_TTL_SECONDS)


def _hermes_home() -> Path:
    """The Hermes home of the current request.

    One gateway process serves several profiles and marks the active one with
    a context-local override (``hermes_constants.get_hermes_home``), while
    ``HERMES_HOME`` stays the root. Asking Hermes keeps a profile's snapshots
    in that profile's cache whether the turn runs in the shared gateway or in
    a process of its own.
    """
    constants = sys.modules.get("hermes_constants")
    if constants is not None:
        with contextlib.suppress(Exception):
            return Path(constants.get_hermes_home())
    home = os.environ.get("HERMES_HOME", "").strip()
    return Path(os.path.expanduser(home)) if home else Path.home() / ".hermes"


def default_store_dir() -> Path:
    """Where snapshots live: env override, else under the Hermes home."""
    override = os.environ.get("ANTIGRAVITY_SESSION_STORE_DIR", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return _hermes_home() / "cache" / _STORE_DIRNAME


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
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=5.0)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _write_private_json(path: Path, payload: Any) -> None:
    """Atomic 0600 JSON write: temp file in the same directory, fsync, rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _safe_name(conversation_id: str) -> str:
    """File name for a conversation id (agy ids are UUIDs; anything else is hashed)."""
    if conversation_id and all(c.isalnum() or c in "-_" for c in conversation_id) and len(conversation_id) <= 128:
        return conversation_id
    return hashlib.sha256(conversation_id.encode("utf-8", errors="replace")).hexdigest()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists, not ours to signal
    return True


class _Claim:
    """An in-process claim: owner reference, claim time, open lock descriptor."""

    __slots__ = ("ref", "since", "fd")

    def __init__(self, ref: weakref.ref, since: float, fd: int | None):
        self.ref = ref
        self.since = since
        self.fd = fd

    def close(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)  # closing the descriptor drops the flock


class SessionRegistry:
    def __init__(
        self,
        *,
        store_dir: Path | str | None = None,
        max_per_key: int | None = None,
        max_total: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        # Reentrant: the weakref callback that frees a dead owner's claim can
        # run from the garbage collector on a thread already holding it.
        self._lock = threading.RLock()
        self._claims: dict[str, _Claim] = {}
        self._fixed_dir = Path(store_dir) if store_dir else None
        self._dir: Path | None = None  # the store the last call used
        self._seen: set[Path] = set()  # stores this process has already swept
        self._max_per_key_arg = max_per_key
        self._max_total_arg = max_total
        self._max_bytes_arg = max_bytes

    # -- limits --------------------------------------------------------------

    @property
    def _max_per_key(self) -> int:
        value = self._max_per_key_arg
        if value is None:
            value = int(_env_float("ANTIGRAVITY_SESSION_MAX_PER_KEY", _MAX_PER_KEY))
        return max(1, value)

    @property
    def _max_total(self) -> int:
        value = self._max_total_arg
        if value is None:
            value = int(_env_float("ANTIGRAVITY_SESSION_MAX_SNAPSHOTS", _MAX_TOTAL))
        return max(1, value)

    @property
    def _max_bytes(self) -> int:
        value = self._max_bytes_arg
        if value is None:
            value = int(_env_float("ANTIGRAVITY_SESSION_MAX_BYTES", _MAX_BYTES))
        return max(1, value)

    # -- storage -------------------------------------------------------------

    def _store(self) -> Path:
        """The private store directory of this request; created 0700, swept on first use.

        Resolved per call (see ``_hermes_home``): one registry may serve
        several profiles in a shared gateway.
        """
        directory = self._fixed_dir or default_store_dir()
        with self._lock:
            self._dir = directory
            first = directory not in self._seen
            self._seen.add(directory)
        if first or not (directory / _CLAIMS_DIRNAME).is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(directory, 0o700)
            (directory / _CLAIMS_DIRNAME).mkdir(mode=0o700, exist_ok=True)
        if first:
            self._sweep_legacy_stores()
            with contextlib.suppress(Exception):
                with self._index_locked(directory) as rows:
                    self._cleanup_locked(rows, time.time(), directory)
        return directory

    @staticmethod
    def _sweep_legacy_stores() -> None:
        """Remove temp directories of the in-memory design whose process is gone."""
        root = Path(tempfile.gettempdir())
        with contextlib.suppress(OSError):
            for entry in root.glob(f"{_LEGACY_STORE_PREFIX}*"):
                pid_text = entry.name[len(_LEGACY_STORE_PREFIX):].split("_", 1)[0]
                if not pid_text.isdigit() or int(pid_text) == os.getpid():
                    continue
                if not _pid_alive(int(pid_text)):
                    shutil.rmtree(entry, ignore_errors=True)

    @contextlib.contextmanager
    def _file_lock(self, path: Path) -> Iterator[None]:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # also drops the flock

    def _read_index(self, directory: Path) -> dict[str, dict[str, Any]]:
        path = directory / _INDEX_NAME
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return {}
        except OSError:
            logger.warning("Antigravity session index %s is unreadable; treating it as empty", path)
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
                raise ValueError("unknown index format")
            rows = data.get("snapshots")
            if not isinstance(rows, dict):
                raise ValueError("snapshots is not an object")
            clean: dict[str, dict[str, Any]] = {}
            for sid, row in rows.items():
                if (
                    isinstance(row, dict)
                    and isinstance(row.get("key"), str)
                    and isinstance(row.get("conversation_id"), str)
                    and isinstance(row.get("history_len"), int)
                    and isinstance(row.get("last_used"), (int, float))
                ):
                    clean[str(sid)] = row
            return clean
        except (ValueError, TypeError):
            logger.warning(
                "Antigravity session index %s is corrupt; treating it as empty "
                "(turns fall back to full sends until the next snapshot)",
                path,
            )
            return {}

    @contextlib.contextmanager
    def _index_locked(self, directory: Path | None = None) -> Iterator[dict[str, dict[str, Any]]]:
        """Read-modify-write of the index under the cross-process lock.

        The yielded dict is written back (atomically) only when it changed.
        """
        directory = directory or self._store()
        with self._lock, self._file_lock(directory / _INDEX_LOCK_NAME):
            rows = self._read_index(directory)
            before = json.dumps(rows, sort_keys=True)
            yield rows
            if json.dumps(rows, sort_keys=True) != before:
                _write_private_json(
                    directory / _INDEX_NAME, {"version": _INDEX_VERSION, "snapshots": rows}
                )

    def _paths(self, snapshot_id: str, directory: Path | None = None) -> tuple[Path, Path]:
        directory = directory or self._store()
        return directory / f"{snapshot_id}.db", directory / f"{snapshot_id}.json"

    def _delete_files(self, snapshot_id: str, directory: Path) -> None:
        db_path, meta_path = self._paths(snapshot_id, directory)
        for path in (meta_path, *(Path(f"{db_path}{s}") for s in ("", "-wal", "-shm"))):
            with contextlib.suppress(OSError):
                path.unlink()

    def _cleanup_locked(self, rows: dict[str, dict[str, Any]], now: float, directory: Path) -> None:
        """TTL, per-key, total and byte caps, orphan files. Caller holds the index lock."""
        ttl = session_ttl_seconds()
        doomed = {sid for sid, row in rows.items() if now - float(row["last_used"]) > ttl}
        live = sorted(
            (item for item in rows.items() if item[0] not in doomed),
            key=lambda item: float(item[1]["last_used"]),
            reverse=True,
        )
        per_key: dict[str, int] = {}
        kept: list[tuple[str, dict[str, Any]]] = []
        total_bytes = 0
        for sid, row in live:
            count = per_key.get(row["key"], 0)
            size = int(row.get("bytes") or 0)
            if (
                count >= self._max_per_key
                or len(kept) >= self._max_total
                or (kept and total_bytes + size > self._max_bytes)
            ):
                doomed.add(sid)
                continue
            per_key[row["key"]] = count + 1
            total_bytes += size
            kept.append((sid, row))
        for sid in doomed:
            rows.pop(sid, None)
            self._delete_files(sid, directory)
        self._sweep_orphans_locked(rows, now, directory)

    def _sweep_orphans_locked(self, rows: dict[str, dict[str, Any]], now: float, directory: Path) -> None:
        with contextlib.suppress(OSError):
            for entry in directory.iterdir():
                name = entry.name
                if name in (_INDEX_NAME, _INDEX_LOCK_NAME, _CLAIMS_DIRNAME):
                    continue
                stem = name.split(".", 1)[0] if not name.startswith(".") else ""
                if stem and stem in rows:
                    continue
                try:
                    age = now - entry.lstat().st_mtime
                except OSError:
                    continue
                if age > _ORPHAN_GRACE_SECONDS and entry.is_file():
                    with contextlib.suppress(OSError):
                        entry.unlink()
        claims = directory / _CLAIMS_DIRNAME
        with contextlib.suppress(OSError):
            for entry in claims.glob("*.lock"):
                self._remove_stale_claim_file(entry, now)

    def _remove_stale_claim_file(self, path: Path, now: float) -> None:
        """Unlink a claim file nobody holds (taken under its own lock, inode checked)."""
        if fcntl is None:
            return
        try:
            if now - path.stat().st_mtime < _CLAIM_MAX_SECONDS:
                return
            fd = os.open(path, os.O_RDWR)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(fd).st_ino == path.stat().st_ino:
                path.unlink()
        except OSError:
            pass
        finally:
            os.close(fd)

    def _load(self, snapshot_id: str, row: dict[str, Any], directory: Path) -> SessionSnapshot | None:
        db_path, meta_path = self._paths(snapshot_id, directory)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            history = meta["history"]
            if not isinstance(history, list) or meta.get("conversation_id") != row["conversation_id"]:
                raise ValueError("snapshot metadata does not match its index row")
            baseline = meta.get("usage_baseline") or {}
            return SessionSnapshot(
                snapshot_id=snapshot_id,
                key=row["key"],
                conversation_id=row["conversation_id"],
                history=history,
                last_assistant=meta.get("last_assistant"),
                usage_baseline={str(k): int(v) for k, v in dict(baseline).items()},
                db_path=db_path,
                last_used=float(row["last_used"]),
            )
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Antigravity session snapshot %s is unreadable; skipping it", snapshot_id)
            return None

    def shutdown(self) -> None:
        """Release this process's claims. Snapshots stay on disk for later processes."""
        with self._lock:
            claims, self._claims = self._claims, {}
        for claim in claims.values():
            claim.close()

    def clear(self) -> None:
        """Delete every snapshot in the store."""
        directory = self._store()
        with self._index_locked(directory) as rows:
            for sid in list(rows):
                rows.pop(sid)
                self._delete_files(sid, directory)

    # -- index ---------------------------------------------------------------

    def __len__(self) -> int:
        try:
            directory = self._store()
            with self._lock, self._file_lock(directory / _INDEX_LOCK_NAME):
                return len(self._read_index(directory))
        except OSError:
            return 0

    @property
    def _snapshots(self) -> dict[str, SessionSnapshot]:
        """Every readable snapshot, loaded (diagnostics and tests)."""
        out: dict[str, SessionSnapshot] = {}
        directory = self._store()
        with self._index_locked(directory) as rows:
            items = list(rows.items())
        for sid, row in items:
            snap = self._load(sid, row, directory)
            if snap is not None:
                out[sid] = snap
        return out

    def touch(self, snapshot_id: str, last_used: float) -> None:
        """Set a snapshot's last use (tests and diagnostics)."""
        with self._index_locked(self._store()) as rows:
            if snapshot_id in rows:
                rows[snapshot_id]["last_used"] = float(last_used)

    # -- claims --------------------------------------------------------------

    def _claim_path(self, conversation_id: str) -> Path:
        return self._store() / _CLAIMS_DIRNAME / f"{_safe_name(conversation_id)}.lock"

    def _local_holder_locked(self, conversation_id: str, now: float) -> Any:
        """Live in-process owner of a claim, pruning dead or expired ones."""
        claim = self._claims.get(conversation_id)
        if claim is None:
            return None
        owner = claim.ref()
        if owner is None or now - claim.since > _CLAIM_MAX_SECONDS:
            del self._claims[conversation_id]
            claim.close()
            return None
        return owner

    def _try_lock_claim_file(self, conversation_id: str) -> tuple[bool, int | None]:
        """(acquired, fd). Acquired without an fd means no cross-process locking."""
        if fcntl is None:
            return True, None
        path = self._claim_path(conversation_id)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        for _ in range(3):
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    raise
                return self._claim_file_is_stale(path), None
            try:
                same = os.fstat(fd).st_ino == path.stat().st_ino
            except OSError:
                same = False
            if not same:  # a sweeper unlinked the file between open and lock
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({"pid": os.getpid(), "since": time.time()}).encode())
            return True, fd
        return False, None

    @staticmethod
    def _claim_file_is_stale(path: Path) -> bool:
        """A held claim older than the bound is ignored (its holder missed a release)."""
        try:
            info = json.loads(path.read_text(encoding="utf-8") or "{}")
            since = float(info.get("since") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            return False
        if since and time.time() - since > _CLAIM_MAX_SECONDS:
            logger.warning(
                "Antigravity conversation claim %s held by pid %s for over %.0fs; ignoring it",
                path.stem,
                info.get("pid"),
                _CLAIM_MAX_SECONDS,
            )
            return True
        return False

    def _owner_gone(self, conversation_id: str, ref: weakref.ref) -> None:
        with self._lock:
            claim = self._claims.get(conversation_id)
            if claim is not None and claim.ref is ref:
                del self._claims[conversation_id]
                claim.close()

    def _claim_locked(self, conversation_id: str, owner: Any, now: float) -> bool:
        holder = self._local_holder_locked(conversation_id, now)
        if holder is owner:
            return True
        if holder is not None:
            return False
        acquired, fd = self._try_lock_claim_file(conversation_id)
        if not acquired:
            return False
        self._drop_owner_claims_locked(owner)
        ref = weakref.ref(owner, lambda r, cid=conversation_id: self._owner_gone(cid, r))
        self._claims[conversation_id] = _Claim(ref, now, fd)
        return True

    def claim(self, conversation_id: str, owner: Any) -> bool:
        """Mark ``conversation_id`` as being used by ``owner``; False if anyone else has it.

        "Anyone" includes other processes sharing the store. Idempotent for the
        same owner. An owner holds one claim at a time.
        """
        if not conversation_id:
            return False
        with self._lock:
            return self._claim_locked(conversation_id, owner, time.monotonic())

    def _drop_owner_claims_locked(self, owner: Any) -> None:
        for cid in [c for c, claim in self._claims.items() if claim.ref() is owner]:
            self._claims.pop(cid).close()

    def release_owner(self, owner: Any) -> None:
        with self._lock:
            if self._claims:
                self._drop_owner_claims_locked(owner)

    def in_use(self, conversation_id: str) -> bool:
        with self._lock:
            if self._local_holder_locked(conversation_id, time.monotonic()) is not None:
                return True
            acquired, fd = self._try_lock_claim_file(conversation_id)
        if fd is not None:
            os.close(fd)
        return not acquired

    # -- snapshots -----------------------------------------------------------

    def drop(self, snapshot_id: str | None) -> None:
        """Forget a snapshot (e.g. one whose conversation agy refused to resume)."""
        if not snapshot_id:
            return
        directory = self._store()
        with self._index_locked(directory) as rows:
            rows.pop(snapshot_id, None)
            self._delete_files(snapshot_id, directory)

    def lookup(
        self, key: str, messages: Sequence[dict[str, Any]], owner: Any = None
    ) -> SessionSnapshot | None:
        """Longest snapshot of ``key`` whose history ``messages`` strictly extend.

        With ``owner``, snapshots of a conversation another client (in any
        process) is using are skipped and the chosen conversation is claimed for
        ``owner`` before it is returned, so two requests cannot both adopt it.
        Never raises on a damaged store: that is "no snapshot".
        """
        try:
            return self._lookup(key, messages, owner)
        except Exception:
            logger.warning("Antigravity session lookup failed; sending a full send", exc_info=True)
            return None

    def _lookup(
        self, key: str, messages: Sequence[dict[str, Any]], owner: Any
    ) -> SessionSnapshot | None:
        directory = self._store()
        now = time.time()
        ttl = session_ttl_seconds()
        with self._index_locked(directory) as rows:
            expired = [s for s, r in rows.items() if now - float(r["last_used"]) > ttl]
            for sid in expired:
                rows.pop(sid)
                self._delete_files(sid, directory)
            candidates = sorted(
                ((sid, dict(row)) for sid, row in rows.items() if row["key"] == key),
                key=lambda item: int(item[1]["history_len"]),
                reverse=True,
            )
        n_messages = len(messages)
        for sid, row in candidates:
            if int(row["history_len"]) >= n_messages:
                continue  # cannot be strictly extended by this request
            snap = self._load(sid, row, directory)
            if snap is None:
                continue
            if not _messages_match_prefix(snap.history, messages):
                continue
            if owner is not None:
                with self._lock:
                    if not self._claim_locked(snap.conversation_id, owner, time.monotonic()):
                        continue
            with self._index_locked(directory) as rows:
                if sid not in rows:  # evicted meanwhile by another process
                    if owner is not None:
                        self.release_owner(owner)
                    continue
                rows[sid]["last_used"] = now
            snap.last_used = now
            return snap
        return None

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
            with contextlib.suppress(Exception):
                self.drop(snap.snapshot_id)
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
        files are swapped in place by rename, so an adopter mid-copy keeps the
        old inode. Never a snapshot taken from another client: those stay
        immutable.
        """
        source_db = Path(source_db)
        if not conversation_id or not source_db.is_file():
            return None
        store = self._store()
        with self._index_locked(store) as rows:
            existing = rows.get(replace) if replace else None
        snapshot_id = replace if (replace and existing is not None) else uuid.uuid4().hex
        db_path, meta_path = self._paths(snapshot_id, store)
        staging = store / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
        try:
            _backup_sqlite(source_db, staging)
            os.replace(staging, db_path)
            _write_private_json(
                meta_path,
                {
                    "version": _INDEX_VERSION,
                    "snapshot_id": snapshot_id,
                    "conversation_id": conversation_id,
                    "history": slim_messages(history),
                    "last_assistant": copy.deepcopy(last_assistant),
                    "usage_baseline": dict(usage_baseline or {}),
                },
            )
            size = db_path.stat().st_size + meta_path.stat().st_size
        except (sqlite3.Error, OSError, TypeError, ValueError):
            logger.debug("Session snapshot of %s failed", source_db, exc_info=True)
            for path in (staging, *(() if existing is not None else (db_path, meta_path))):
                with contextlib.suppress(OSError):
                    Path(path).unlink()
            return None
        now = time.time()
        with self._index_locked(store) as rows:
            rows[snapshot_id] = {
                "key": key,
                "conversation_id": conversation_id,
                "history_len": len(history),
                "last_used": now,
                "created": now,
                "bytes": size,
                "pid": os.getpid(),
            }
            self._cleanup_locked(rows, now, store)
            kept = snapshot_id in rows
        if kept:
            logger.debug(
                "Antigravity session snapshot %s published: conversation %s, %d messages, %d bytes",
                snapshot_id,
                conversation_id,
                len(history),
                size,
            )
        return snapshot_id if kept else None


_REGISTRY = SessionRegistry()


def get_registry() -> SessionRegistry:
    return _REGISTRY
