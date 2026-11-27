"""Session snapshots survive a restart and are shared safely between processes.

Hermes loads this plugin in several processes at once (gateway, dashboard,
cron, CLI) and restarts them on updates. The registry therefore keeps its
snapshots and index on disk, guards the index with ``flock`` and turns the
"one in-flight turn per conversation" claim into a per-conversation lock file.
These tests pin: a second registry on the same directory (a new process) finds
what the first one published; claims exclude each other across real processes;
a dead holder's claim is taken over; TTL, count and byte caps; 0600/0700
permissions; and a damaged store degrading to a full send without raising.
"""

import multiprocessing
import os
import signal
import sqlite3
import stat
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _registry_child  # noqa: E402
import client as client_module  # noqa: E402
import session_registry  # noqa: E402
from session_registry import SessionRegistry, default_store_dir  # noqa: E402
from test_session_resume import (  # noqa: E402,F401
    A1,
    SYSTEM,
    U1,
    U2,
    U3,
    A2,
    _is_full_send,
    _turn,
    agy,
    make_client,
    registry,
)

posix_only = pytest.mark.skipif(session_registry.fcntl is None, reason="needs fcntl")
_CTX = multiprocessing.get_context("spawn")


class Owner:
    pass


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "src" / "c1.db"
    path.parent.mkdir()
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return path


def _seed(reg, key, history, *, source, conv="c1"):
    return reg.publish(
        key,
        conversation_id=conv,
        history=history,
        last_assistant=None,
        usage_baseline={"input_tokens": 7},
        source_db=source,
    )


# -- location -----------------------------------------------------------------


def test_default_store_dir_follows_hermes_home_and_the_override(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.delenv("ANTIGRAVITY_SESSION_STORE_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_constants", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    assert default_store_dir() == tmp_path / "hh" / "cache" / "agy-session-snapshots"
    monkeypatch.delenv("HERMES_HOME")
    assert default_store_dir() == Path.home() / ".hermes" / "cache" / "agy-session-snapshots"
    # Inside Hermes the profile of the current request wins over HERMES_HOME:
    # a shared gateway keeps HERMES_HOME at the root and marks the profile
    # with a context-local override.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root"))
    profile = tmp_path / "root" / "profiles" / "p1"
    monkeypatch.setitem(sys.modules, "hermes_constants", SimpleNamespace(get_hermes_home=lambda: profile))
    assert default_store_dir() == profile / "cache" / "agy-session-snapshots"
    monkeypatch.setenv("ANTIGRAVITY_SESSION_STORE_DIR", str(tmp_path / "elsewhere"))
    assert default_store_dir() == tmp_path / "elsewhere"


def test_one_registry_serves_each_profile_from_its_own_store(monkeypatch, tmp_path, source_db):
    from types import SimpleNamespace

    monkeypatch.delenv("ANTIGRAVITY_SESSION_STORE_DIR", raising=False)
    current = {"home": tmp_path / "p1"}
    monkeypatch.setitem(sys.modules, "hermes_constants", SimpleNamespace(get_hermes_home=lambda: current["home"]))
    reg = SessionRegistry()
    _seed(reg, "k", [U1], source=source_db)
    current["home"] = tmp_path / "p2"
    assert reg.lookup("k", [U1, A1]) is None
    current["home"] = tmp_path / "p1"
    assert reg.lookup("k", [U1, A1]) is not None
    assert (tmp_path / "p1" / "cache" / "agy-session-snapshots" / "index.json").is_file()


# -- 1. a new process finds what an earlier one published ----------------------


def test_second_registry_on_the_same_directory_finds_the_snapshot(tmp_path, source_db):
    store = tmp_path / "store"
    first = SessionRegistry(store_dir=store)
    sid = _seed(first, "k", [U1], source=source_db)
    first.shutdown()  # the first process is gone; nothing is deleted

    second = SessionRegistry(store_dir=store)  # a new process, same directory
    snap = second.lookup("k", [U1, A1, U2])
    assert snap is not None
    assert (snap.snapshot_id, snap.conversation_id, snap.history) == (sid, "c1", [
        {"role": "user", "content": "turn one", "tool_calls": None}
    ])
    assert snap.usage_baseline == {"input_tokens": 7}
    dest = tmp_path / "home2" / "conversations"
    assert second.restore(snap, dest)
    assert sqlite3.connect(dest / "c1.db").execute("SELECT x FROM t").fetchall() == [(1,)]


def test_a_real_second_process_finds_the_snapshot(tmp_path, source_db):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    _seed(reg, "k", [U1, A1, U2], source=source_db)
    result = _CTX.Queue()
    proc = _CTX.Process(
        target=_registry_child.lookup_history_len,
        args=(store, "k", [U1, A1, U2, A2, U3], result),
    )
    proc.start()
    proc.join(60)
    assert result.get(timeout=5) == ("c1", 3)


def test_restarted_process_resumes_the_conversation_with_only_the_delta(agy, make_client, tmp_path):
    store = tmp_path / "persist"
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=store)):
        a = make_client()
        _turn(a, [SYSTEM, U1])
        a.close()
    # Process restart: a brand-new registry object reading the same directory.
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=store)):
        b = make_client()
        _turn(b, [SYSTEM, U1, A1, U2])
    first, second = agy.spawns
    assert second["args"][second["args"].index("--conversation") + 1] == first["conv"]
    assert second["rows_before"] == 1
    sent = agy.prompt_sent(1)
    assert "turn two" in sent and "turn one" not in sent


def test_worker_send_log_tells_delta_from_full_send(agy, make_client, tmp_path, caplog):
    store = tmp_path / "persist"
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=store)):
        with caplog.at_level("INFO", logger="client"):
            a = make_client()
            _turn(a, [SYSTEM, U1])
            a.close()
            b = make_client()
            _turn(b, [SYSTEM, U1, A1, U2])
    sends = [r.getMessage() for r in caplog.records if "worker send" in r.getMessage()]
    conv = agy.spawns[0]["conv"]
    assert f"pid={os.getpid()} branch=full conversation=- messages=2/2" in sends[0]
    # A1 differs from what the scripted agy answered, so it is echoed with U2.
    assert f"branch=delta conversation={conv} messages=2/4" in sends[1]
    adopted = [r.getMessage() for r in caplog.records if "adopted conversation" in r.getMessage()]
    assert adopted and f"pid={os.getpid()} adopted conversation {conv}" in adopted[0]


# -- 2. claims exclude each other across processes ------------------------------


@posix_only
def test_claim_held_by_another_process_excludes_this_one(tmp_path, source_db):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    _seed(reg, "k", [U1], source=source_db, conv="c1")
    claimed, release, result = _CTX.Event(), _CTX.Event(), _CTX.Queue()
    child = _CTX.Process(target=_registry_child.hold_claim, args=(store, "c1", claimed, release, result))
    child.start()
    try:
        assert claimed.wait(60)
        child_pid, ok = result.get(timeout=5)
        assert ok and child_pid != os.getpid()
        me = Owner()
        assert reg.in_use("c1")
        assert not reg.claim("c1", me)
        assert reg.lookup("k", [U1, A1], owner=me) is None  # cannot adopt it either
        assert reg.lookup("k", [U1, A1]) is not None  # read-only lookups still see it
    finally:
        release.set()
        child.join(30)
    assert not reg.in_use("c1")
    assert reg.claim("c1", me)
    reg.shutdown()


@posix_only
def test_simultaneous_claims_from_many_processes_have_one_winner(tmp_path):
    store = tmp_path / "store"
    SessionRegistry(store_dir=store)._store()
    start, done, result = _CTX.Event(), _CTX.Event(), _CTX.Queue()
    children = [
        _CTX.Process(target=_registry_child.race_claim, args=(store, "c1", start, done, result))
        for _ in range(6)
    ]
    for child in children:
        child.start()
    time.sleep(1.0)  # let every child import and park on the event
    start.set()
    outcomes = [result.get(timeout=60) for _ in children]
    done.set()
    for child in children:
        child.join(30)
    assert sorted(outcomes) == [False] * 5 + [True]


# -- 3. a dead holder's claim is taken over -------------------------------------


@posix_only
def test_claim_of_a_killed_process_is_taken_over(tmp_path):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    claimed, release, result = _CTX.Event(), _CTX.Event(), _CTX.Queue()
    child = _CTX.Process(target=_registry_child.hold_claim, args=(store, "c1", claimed, release, result))
    child.start()
    assert claimed.wait(60)
    assert result.get(timeout=5)[1]
    me = Owner()
    assert not reg.claim("c1", me)
    os.kill(child.pid, signal.SIGKILL)  # no release, no atexit, no finally
    child.join(30)
    assert reg.claim("c1", me)
    reg.shutdown()


@posix_only
def test_claim_older_than_the_bound_is_ignored(tmp_path, monkeypatch):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    claimed, release, result = _CTX.Event(), _CTX.Event(), _CTX.Queue()
    child = _CTX.Process(target=_registry_child.hold_claim, args=(store, "c1", claimed, release, result))
    child.start()
    try:
        assert claimed.wait(60)
        result.get(timeout=5)
        monkeypatch.setattr(session_registry, "_CLAIM_MAX_SECONDS", 0.0)
        time.sleep(0.05)
        assert reg.claim("c1", Owner())
    finally:
        release.set()
        child.join(30)
    reg.shutdown()


def test_claim_is_freed_when_its_owner_is_collected(tmp_path):
    import gc

    reg = SessionRegistry(store_dir=tmp_path / "store")
    a = Owner()
    assert reg.claim("c1", a)
    del a
    gc.collect()
    assert not reg.in_use("c1")
    assert reg.claim("c1", Owner())


# -- 4. clean-up ------------------------------------------------------------------


def test_ttl_count_and_byte_caps(tmp_path, source_db, monkeypatch):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store, max_total=3)
    ids = []
    for i in range(5):
        ids.append(_seed(reg, f"k{i}", [U1], source=source_db))
        time.sleep(0.01)
    assert set(reg._snapshots) == set(ids[2:])
    assert sorted(p.name for p in store.glob("*.db")) == sorted(f"{s}.db" for s in ids[2:])

    # TTL: evaluated by a fresh registry (a new process) at start-up.
    reg.touch(ids[2], time.time() - 7200)
    monkeypatch.setenv("ANTIGRAVITY_SESSION_TTL_SECONDS", "3600")
    fresh = SessionRegistry(store_dir=store, max_total=3)
    assert set(fresh._snapshots) == set(ids[3:])
    assert not (store / f"{ids[2]}.db").exists() and not (store / f"{ids[2]}.json").exists()

    # Bytes: room for exactly two snapshots; the newest always stays.
    size = fresh._snapshots[ids[3]].db_path.stat().st_size + (store / f"{ids[3]}.json").stat().st_size
    small = SessionRegistry(store_dir=store, max_total=10, max_bytes=2 * size + size // 2)
    newest = _seed(small, "k9", [U1], source=source_db)
    assert set(small._snapshots) == {ids[4], newest}


def test_byte_cap_from_the_environment(tmp_path, source_db, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_SESSION_MAX_BYTES", "1")
    monkeypatch.setenv("ANTIGRAVITY_SESSION_MAX_SNAPSHOTS", "50")
    reg = SessionRegistry(store_dir=tmp_path / "store")
    assert reg._max_bytes == 1 and reg._max_total == 50
    _seed(reg, "a", [U1], source=source_db)
    last = _seed(reg, "b", [U1], source=source_db)
    assert set(reg._snapshots) == {last}  # over the cap, only the newest survives


def test_orphan_files_and_stale_temp_files_are_swept(tmp_path, source_db):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    keep = _seed(reg, "k", [U1], source=source_db)
    old = time.time() - 3600
    orphan = store / "deadbeef.db"
    orphan.write_bytes(b"x")
    tmp = store / ".abc.tmp"
    tmp.write_bytes(b"x")
    young = store / "cafe.db"
    young.write_bytes(b"x")  # may be a publish in flight elsewhere
    for path in (orphan, tmp):
        os.utime(path, (old, old))
    _seed(reg, "k2", [U1], source=source_db)
    assert not orphan.exists() and not tmp.exists()
    assert young.exists()
    assert (store / f"{keep}.db").exists()


# -- 5. permissions ---------------------------------------------------------------


@posix_only
def test_store_directory_and_files_are_private(tmp_path, source_db):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    sid = _seed(reg, "k", [U1], source=source_db)
    assert reg.claim("c1", owner := Owner())
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(store) == 0o700
    assert mode(store / "claims") == 0o700
    for path in (store / f"{sid}.db", store / f"{sid}.json", store / "index.json", store / "index.lock",
                 store / "claims" / "c1.lock"):
        assert mode(path) == 0o600, path
    dest = tmp_path / "dest"
    reg.restore(reg._snapshots[sid], dest)
    assert mode(dest / "c1.db") == 0o600
    reg.release_owner(owner)


# -- 6. a damaged store is a full send, never an error ------------------------------


@pytest.mark.parametrize(
    "garbage",
    [b"{not json", b"[]", b'{"version": 1, "snapshots": {"x": {"key": 5}}}', b'{"version": 99}', b"\xff\xfe"],
)
def test_corrupt_index_is_treated_as_empty(tmp_path, source_db, garbage, caplog):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    _seed(reg, "k", [U1], source=source_db)
    (store / "index.json").write_bytes(garbage)
    fresh = SessionRegistry(store_dir=store)
    assert fresh.lookup("k", [U1, A1], owner=Owner()) is None
    assert len(fresh) == 0
    # The next publish rewrites a valid index.
    sid = _seed(fresh, "k", [U1], source=source_db)
    assert fresh.lookup("k", [U1, A1]).snapshot_id == sid


def test_corrupt_snapshot_metadata_is_skipped(tmp_path, source_db):
    store = tmp_path / "store"
    reg = SessionRegistry(store_dir=store)
    older = _seed(reg, "k", [U1], source=source_db)
    newer = _seed(reg, "k", [U1, A1, U2], source=source_db)
    (store / f"{newer}.json").write_text("{broken")
    snap = reg.lookup("k", [U1, A1, U2, A2, U3])
    assert snap is not None and snap.snapshot_id == older


def test_corrupt_store_makes_the_client_full_send_without_raising(agy, make_client, tmp_path):
    store = tmp_path / "persist"
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=store)):
        a = make_client()
        _turn(a, [SYSTEM, U1])
        a.close()
    (store / "index.json").write_text("{definitely not json")
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=store)):
        b = make_client()
        chunks = _turn(b, [SYSTEM, U1, A1, U2])
    assert chunks
    assert "--conversation" not in agy.spawns[1]["args"]
    assert _is_full_send(agy.prompt_sent(1))


def test_unwritable_store_never_fails_the_request(agy, make_client, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with patch.object(client_module, "get_registry", return_value=SessionRegistry(store_dir=blocker / "s")):
        a = make_client()
        assert _turn(a, [SYSTEM, U1])
        b = make_client()
        assert _turn(b, [SYSTEM, U1, A1, U2])
    assert "--conversation" not in agy.spawns[1]["args"]
