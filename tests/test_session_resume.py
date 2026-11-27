"""Cross-client agy conversation resume (session_registry.py).

Hermes builds a new agent, and with it a new AntigravityClient, for every
gateway message, and runs its background review on yet another one. Before the
registry each of them started with an empty history and re-sent the whole
prompt (6 split parts, ~350k characters measured with a Gemini route). These
tests drive real clients over a scripted agy and pin that a later client of the
same Hermes session restores the conversation file, respawns with
``--conversation <id>`` and sends only the new messages, and that nothing else
(forks, other models, other sessions, unrelated requests) leaks into the line.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import client as client_module  # noqa: E402
import session_registry  # noqa: E402
from client import AntigravityClient  # noqa: E402
from session_registry import SessionRegistry, session_key, slim_messages  # noqa: E402
from test_worker_usage import MODEL, _mock_proc, _turn_events, _turn_lines  # noqa: E402

_REAL_POPEN = subprocess.Popen
SESSION = "20261127_093400_abc123"

U1 = {"role": "user", "content": "turn one"}
A1 = {"role": "assistant", "content": "answer one"}
U2 = {"role": "user", "content": "turn two"}
A2 = {"role": "assistant", "content": "answer two"}
U3 = {"role": "user", "content": "turn three"}
SYSTEM = {"role": "system", "content": "You are the room bot."}


def _usage(inp, out, cached=0):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out, "cache_read_tokens": cached}


class FakeAgy:
    """Popen stand-in for agy that keeps a real SQLite conversation file.

    Every spawn appends one row to ``conversations/<id>.db`` in the HOME it was
    started in, so the row count of a resumed file proves which snapshot was
    restored (and that a fork's rows did not leak into the main line).
    """

    def __init__(self):
        self.spawns: list[dict] = []
        self._next = 0
        # agy answers an unknown --conversation id with a NEW conversation.
        self.lose_resume = False
        # agy that never announces a conversation (dies / hangs at start-up).
        self.silent_resume = False
        # The next spawn's turn ends in an error result.
        self.fail_next = False

    def popen(self, args, *a, **kw):
        if not args or args[0] != "agy":
            return _REAL_POPEN(args, *a, **kw)
        if "--conversation" in args:
            conv = args[args.index("--conversation") + 1]
            if self.lose_resume:
                self._next += 1
                conv = f"lost-{self._next}"
        else:
            self._next += 1
            conv = f"conv-{self._next}"
        conv_dir = Path(kw["cwd"]) / "home" / ".gemini" / "antigravity-cli" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        db = conv_dir / f"{conv}.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE IF NOT EXISTS turns (n INTEGER)")
        rows_before = con.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        con.execute("INSERT INTO turns VALUES (?)", (len(self.spawns),))
        con.commit()
        con.close()
        # A live worker may serve several turns: script three, usage cumulative.
        turns = [
            _turn_events(conv, f"reply {len(self.spawns)}.{i}", _usage(1000 * (i + 1), 10 * (i + 1)))
            for i in range(3)
        ]
        if self.fail_next:
            self.fail_next = False
            turns = [
                [
                    {"event": "init", "conversation_id": conv},
                    {"event": "result", "result": {"status": "ERROR", "error": "boom", "response": ""}},
                ]
            ]
        proc = _mock_proc(_turn_lines(*turns))
        if self.silent_resume and "--conversation" in args:
            proc.stdout.readline.side_effect = [""]
        self.spawns.append(
            {"args": list(args), "conv": conv, "rows_before": rows_before, "cwd": kw["cwd"], "proc": proc}
        )
        return proc

    def prompt_sent(self, index: int) -> str:
        writes = self.spawns[index]["proc"].stdin.write.call_args_list
        return "".join(json.loads(c.args[0])["message"]["content"] for c in writes if c.args[0].strip())


@pytest.fixture
def registry(tmp_path):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    with patch.object(client_module, "get_registry", return_value=reg):
        yield reg
    reg.shutdown()


@pytest.fixture
def agy(registry):
    fake = FakeAgy()
    patches = [
        patch("client.is_authenticated", return_value=True),
        patch("process.resolve_real_token_path", return_value=None),
        patch("client.resolve_agy_command", return_value="agy"),
        patch("process._link_macos_keychains"),
        patch("subprocess.Popen", side_effect=fake.popen),
    ]
    for p in patches:
        p.start()
    yield fake
    for p in patches:
        p.stop()


@pytest.fixture
def make_client(tmp_path):
    made = []

    def make():
        directory = tempfile.mkdtemp(prefix="hermes_agy_test_", dir=tmp_path)
        c = AntigravityClient(cwd=directory)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def _turn(client, messages, *, session=SESSION, model=MODEL, **kw):
    kwargs = dict(model=model, messages=messages, stream=True)
    if session is not None:
        kwargs["hermes_session_id"] = session
    kwargs.update(kw)
    return list(client.chat.completions.create(**kwargs))


def _is_full_send(prompt: str, first_user: str = "turn one") -> bool:
    return first_user in prompt


def test_new_client_resumes_conversation_and_sends_only_the_delta(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()

    b = make_client()
    chunks = _turn(b, [SYSTEM, U1, A1, U2])

    first, second = agy.spawns
    assert "--conversation" not in first["args"]
    assert second["args"][second["args"].index("--conversation") + 1] == first["conv"]
    # The conversation file of the first client was restored into the new HOME.
    assert second["rows_before"] == 1
    assert second["cwd"] != first["cwd"]
    sent = agy.prompt_sent(1)
    assert "turn two" in sent and "turn one" not in sent
    assert chunks  # the turn completed


def test_resume_does_not_need_the_first_client_to_be_closed(agy, make_client):
    # Hermes does not always close a request client; the snapshot is taken at
    # the end of the turn, from the live conversation file.
    a = make_client()
    _turn(a, [SYSTEM, U1])
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    assert agy.spawns[1]["rows_before"] == 1
    assert "--conversation" in agy.spawns[1]["args"]


def test_chain_of_clients_keeps_the_conversation_growing(agy, make_client):
    for i, messages in enumerate(
        ([SYSTEM, U1], [SYSTEM, U1, A1, U2], [SYSTEM, U1, A1, U2, A2, U3])
    ):
        c = make_client()
        _turn(c, messages)
        c.close()
    assert [s["rows_before"] for s in agy.spawns] == [0, 1, 2]
    assert len({s["conv"] for s in agy.spawns}) == 1
    assert "turn one" not in agy.prompt_sent(2) and "turn two" not in agy.prompt_sent(2)


def test_usage_of_a_resumed_turn_is_a_delta_not_the_session_total(agy, make_client):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b = make_client()
    # The scripted resumed worker reports session totals continuing from turn 1.
    agy.spawns  # first spawn done; script the next one with cumulative counters
    original = agy.popen

    def popen(args, *a_, **kw):
        proc = original(args, *a_, **kw)
        if "--conversation" in args:
            proc.stdout.readline.side_effect = _turn_lines(_turn_events("conv-1", "r", _usage(1500, 30)))
        return proc

    with patch("subprocess.Popen", side_effect=popen):
        chunks = _turn(b, [SYSTEM, U1, A1, U2])
    usage = next(c.usage for c in chunks if getattr(c, "usage", None) is not None)
    assert usage.prompt_tokens == 500  # 1500 - the 1000 already in the baseline
    assert usage.completion_tokens == 20


def test_without_a_session_id_every_client_starts_fresh(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1], session=None)
    a.close()
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2], session=None)
    assert "--conversation" not in agy.spawns[1]["args"]
    assert _is_full_send(agy.prompt_sent(1))
    assert len(registry) == 0


def test_other_session_model_effort_or_system_prompt_never_resume(agy, make_client):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    history = [SYSTEM, U1, A1, U2]
    other = [
        dict(session="another-session"),
        dict(model="gemini-3.8-flash-low"),
        dict(reasoning_effort="low"),
    ]
    for kwargs in other:
        c = make_client()
        _turn(c, history, **kwargs)
        c.close()
    c = make_client()
    _turn(c, [{"role": "system", "content": "A different system prompt."}, U1, A1, U2])
    for i in range(1, 5):
        assert "--conversation" not in agy.spawns[i]["args"], i
        assert agy.spawns[i]["rows_before"] == 0, i


def test_other_account_never_resumes(agy, make_client):
    with patch.object(AntigravityClient, "_resolve_real_token_path", return_value=Path("/accounts/a/token")):
        a = make_client()
        _turn(a, [SYSTEM, U1])
        a.close()
    with patch.object(AntigravityClient, "_resolve_real_token_path", return_value=Path("/accounts/b/token")):
        b = make_client()
        _turn(b, [SYSTEM, U1, A1, U2])
    assert "--conversation" not in agy.spawns[1]["args"]


def test_a_fork_does_not_pollute_the_main_line(agy, make_client, registry):
    # Main turn 1, then Hermes' background review: same session id, same
    # prefix, its own instructions appended. The review must not become the
    # state the next main message continues from.
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()

    review = make_client()
    _turn(review, [SYSTEM, U1, A1, {"role": "user", "content": "REVIEW the conversation above"}])
    review.close()
    assert agy.spawns[1]["rows_before"] == 1  # the review itself resumed cheaply

    main = make_client()
    _turn(main, [SYSTEM, U1, A1, U2])
    main.close()
    # Resumed from the turn-1 snapshot (1 row), not from the review's (2 rows).
    assert agy.spawns[2]["rows_before"] == 1
    assert "REVIEW" not in agy.prompt_sent(2)
    assert "turn two" in agy.prompt_sent(2)

    # And the main line keeps going from its own newest snapshot.
    nxt = make_client()
    _turn(nxt, [SYSTEM, U1, A1, U2, A2, U3])
    assert agy.spawns[3]["rows_before"] == 2
    assert "turn two" not in agy.prompt_sent(3)


def test_two_clients_can_adopt_the_same_snapshot_concurrently(agy, make_client):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b, c = make_client(), make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    _turn(c, [SYSTEM, U1, A1, {"role": "user", "content": "side question"}])
    assert [agy.spawns[1]["rows_before"], agy.spawns[2]["rows_before"]] == [1, 1]
    assert agy.spawns[1]["cwd"] != agy.spawns[2]["cwd"]


def test_unrelated_request_is_a_full_send_and_leaves_the_line_intact(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    before = len(registry)

    x = make_client()
    _turn(x, [SYSTEM, {"role": "user", "content": "something else entirely"}])
    x.close()
    assert "--conversation" not in agy.spawns[1]["args"]
    assert len(registry) == before + 1

    # The original line still resumes afterwards.
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    assert agy.spawns[2]["args"][agy.spawns[2]["args"].index("--conversation") + 1] == agy.spawns[0]["conv"]
    assert agy.spawns[2]["rows_before"] == 1


def test_context_rewrite_such_as_compression_is_a_full_send(agy, make_client):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b = make_client()
    summary = {"role": "user", "content": "[summary of earlier turns]"}
    _turn(b, [SYSTEM, summary, A1, U2])
    assert "--conversation" not in agy.spawns[1]["args"]


def test_same_client_continuation_never_consults_the_registry(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    with patch.object(registry, "lookup", side_effect=AssertionError("must not be called")):
        _turn(a, [SYSTEM, U1, A1, U2])
    assert len(agy.spawns) == 1  # the live worker took the delta


def test_snapshot_whose_file_vanished_falls_back_to_a_full_send(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    for snap in list(registry._snapshots.values()):
        snap.db_path.unlink()
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    assert "--conversation" not in agy.spawns[1]["args"]
    assert _is_full_send(agy.prompt_sent(1))
    assert agy.spawns[1]["rows_before"] == 0


def test_close_refreshes_the_clients_own_snapshot_instead_of_adding_one(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    assert len(registry) == 1
    snapshot_id = a._session_snapshot_id
    a.close()
    assert len(registry) == 1
    assert a._session_snapshot_id == snapshot_id
    assert set(registry._snapshots) == {snapshot_id}
    assert registry._snapshots[snapshot_id].db_path.is_file()


def test_adopted_parent_is_not_refreshed_by_the_adopter(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    parent_ids = set(registry._snapshots)
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    b.close()
    assert parent_ids < set(registry._snapshots)  # parent kept, child added


def test_kill_switch_disables_publish_and_resume(agy, make_client, registry, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_SESSION_RESUME", "0")
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b = make_client()
    _turn(b, [SYSTEM, U1, A1, U2])
    assert len(registry) == 0
    assert "--conversation" not in agy.spawns[1]["args"]


def test_registry_failure_never_fails_the_request(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b = make_client()
    with patch.object(registry, "lookup", side_effect=RuntimeError("boom")):
        chunks = _turn(b, [SYSTEM, U1, A1, U2])
    assert chunks
    assert "--conversation" not in agy.spawns[1]["args"]


def test_oneshot_requests_do_not_touch_the_registry(agy, make_client, registry):
    a = make_client()
    a._worker_lock.acquire()  # a busy worker pushes the next request onto the oneshot path
    try:
        _turn(a, [SYSTEM, U1])
    finally:
        a._worker_lock.release()
    assert len(registry) == 0


# -- registry unit tests -------------------------------------------------------


def _seed(reg, key, n, *, source, conv="c1", history=None):
    return reg.publish(
        key,
        conversation_id=conv,
        history=history or [U1] * n,
        last_assistant=None,
        usage_baseline={},
        source_db=source,
    )


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


def test_session_key_separates_everything_that_must_not_share_a_conversation():
    base = dict(session_id="s", model="m", effort="high", account="a", messages=[SYSTEM, U1])
    k = session_key(**base)
    assert k == session_key(**base)
    assert k != session_key(**{**base, "session_id": "t"})
    assert k != session_key(**{**base, "model": "n"})
    assert k != session_key(**{**base, "effort": "low"})
    assert k != session_key(**{**base, "account": "b"})
    assert k != session_key(**{**base, "messages": [{"role": "system", "content": "x"}, U1]})
    assert session_key(**{**base, "session_id": None}) is None
    assert session_key(**{**base, "session_id": ""}) is None
    # Later messages are not part of the key: that is what lets a line grow.
    assert k == session_key(**{**base, "messages": [SYSTEM, U1, A1, U2]})


def test_slim_messages_copies_only_matched_fields_and_does_not_alias():
    original = [{"role": "assistant", "content": [{"type": "text", "text": "x"}], "tool_calls": [{"id": "1"}], "reasoning": "r"}]
    slim = slim_messages(original)
    assert set(slim[0]) == {"role", "content", "tool_calls"}
    original[0]["content"][0]["text"] = "mutated"
    original[0]["tool_calls"][0]["id"] = "2"
    assert slim[0]["content"][0]["text"] == "x" and slim[0]["tool_calls"][0]["id"] == "1"


def test_publish_copies_a_live_wal_database_consistently(tmp_path):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    live = tmp_path / "live.db"
    writer = sqlite3.connect(live)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE t (x)")
    writer.execute("INSERT INTO t VALUES (7)")
    writer.commit()  # still open: the data may sit in the -wal file only
    sid = _seed(reg, "k", 1, source=live)
    snap = reg._snapshots[sid]
    assert sqlite3.connect(snap.db_path).execute("SELECT x FROM t").fetchall() == [(7,)]
    writer.close()
    reg.shutdown()


def test_publish_without_a_source_file_or_with_garbage_returns_none(tmp_path):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    assert _seed(reg, "k", 1, source=tmp_path / "missing.db") is None
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"not a database" * 100)
    assert _seed(reg, "k", 1, source=garbage) is None
    assert len(reg) == 0
    assert not list((tmp_path / "store").glob("*.db"))
    reg.shutdown()


def test_lookup_returns_the_longest_strictly_extended_history(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    _seed(reg, "k", 0, source=source_db, history=[U1])
    _seed(reg, "k", 0, source=source_db, history=[U1, A1, U2])
    _seed(reg, "other", 0, source=source_db, history=[U1, A1, U2, A2])
    incoming = [U1, A1, U2, A2, U3]
    assert len(reg.lookup("k", incoming).history) == 3
    assert reg.lookup("k", [U1, A1, U2]) is not None and len(reg.lookup("k", [U1, A1, U2]).history) == 1
    assert reg.lookup("k", [U1]) is None  # equal length is not an extension
    assert reg.lookup("k", [U2, A1]) is None
    assert reg.lookup("missing", incoming) is None
    reg.shutdown()


def test_per_key_cap_evicts_least_recently_used_and_deletes_files(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store", max_per_key=2)
    first = _seed(reg, "k", 0, source=source_db, history=[U1])
    time.sleep(0.01)
    second = _seed(reg, "k", 0, source=source_db, history=[U1, A1])
    time.sleep(0.01)
    reg.lookup("k", [U1, A1, U2])  # touches `second`
    time.sleep(0.01)
    third = _seed(reg, "k", 0, source=source_db, history=[U1, A1, U2])
    assert set(reg._snapshots) == {second, third}
    assert not (tmp_path / "store" / f"{first}.db").exists()
    reg.shutdown()


def test_total_cap_evicts_the_oldest_across_keys(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store", max_total=3)
    ids = []
    for i in range(5):
        ids.append(_seed(reg, f"k{i}", 0, source=source_db, history=[U1]))
        time.sleep(0.01)
    assert set(reg._snapshots) == set(ids[2:])
    assert len(list((tmp_path / "store").glob("*.db"))) == 3
    reg.shutdown()


def test_ttl_expires_idle_snapshots(tmp_path, source_db, monkeypatch):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    sid = _seed(reg, "k", 0, source=source_db, history=[U1])
    monkeypatch.setenv("ANTIGRAVITY_SESSION_TTL_SECONDS", "60")
    reg.touch(sid, time.time() - 61)
    assert reg.lookup("k", [U1, A1]) is None
    assert len(reg) == 0
    assert not (tmp_path / "store" / f"{sid}.db").exists()
    reg.shutdown()


def test_ttl_zero_or_garbage_env_values(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_SESSION_TTL_SECONDS", "0")
    assert session_registry.session_resume_enabled() is False
    monkeypatch.setenv("ANTIGRAVITY_SESSION_TTL_SECONDS", "soon")
    assert session_registry.session_ttl_seconds() == session_registry._DEFAULT_TTL_SECONDS
    assert session_registry.session_resume_enabled() is True
    monkeypatch.delenv("ANTIGRAVITY_SESSION_TTL_SECONDS")
    for off in ("0", "false", "No", "OFF"):
        monkeypatch.setenv("ANTIGRAVITY_SESSION_RESUME", off)
        assert session_registry.session_resume_enabled() is False


def test_store_directory_is_recreated_and_survives_shutdown(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    sid = _seed(reg, "k", 0, source=source_db, history=[U1])
    store = reg._dir
    import shutil

    snap = reg._snapshots[sid]
    shutil.rmtree(store)  # e.g. a manual clean-up
    assert reg.lookup("k", [U1, A1]) is None  # index gone with it
    assert reg.restore(snap, tmp_path / "dest") is False
    assert _seed(reg, "k", 0, source=source_db, history=[U1]) is not None  # and a new one works
    reg.shutdown()
    assert store.is_dir() and len(reg) == 1  # shutdown no longer deletes anything


def test_dead_process_legacy_temp_directories_are_swept(tmp_path, monkeypatch, source_db):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    dead = tmp_path / f"hermes_agy_sessions_{done.pid}_xyz"
    dead.mkdir()
    (dead / "x.db").write_bytes(b"x")
    alive = tmp_path / f"hermes_agy_sessions_{os.getppid()}_other"
    alive.mkdir()
    reg = SessionRegistry(store_dir=tmp_path / "store")
    _seed(reg, "k", 0, source=source_db, history=[U1])
    assert not dead.exists()
    assert alive.exists()
    reg.shutdown()


def test_restore_replaces_a_stale_wal_of_the_same_conversation(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    sid = _seed(reg, "k", 0, source=source_db, history=[U1])
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "c1.db-wal").write_bytes(b"stale")
    (dest / "c1.db-shm").write_bytes(b"stale")
    assert reg.restore(reg._snapshots[sid], dest) is True
    assert not (dest / "c1.db-wal").exists() and not (dest / "c1.db-shm").exists()
    assert sqlite3.connect(dest / "c1.db").execute("SELECT x FROM t").fetchall() == [(1,)]
    reg.shutdown()


# -- provider profile ---------------------------------------------------------


def test_profile_forwards_the_hermes_session_id_to_the_client():
    from conftest import PLUGIN

    profile = PLUGIN.antigravity_profile
    _, top = profile.build_api_kwargs_extras(
        reasoning_config={"effort": "high"}, model=MODEL, session_id="sess-1", cache_scope_id="scope"
    )
    assert top == {"reasoning_effort": "high", "hermes_session_id": "sess-1"}
    _, top = profile.build_api_kwargs_extras(reasoning_config=None, model=MODEL)
    assert "hermes_session_id" not in top
    _, top = profile.build_api_kwargs_extras(reasoning_config=None, model=MODEL, session_id="")
    assert "hermes_session_id" not in top


# -- agy refuses to resume the conversation --------------------------------------


def _texts(chunks):
    return "".join(
        c.choices[0].delta.content or ""
        for c in chunks
        if getattr(c, "choices", None) and getattr(c.choices[0], "delta", None)
    )


@pytest.mark.parametrize("stream", [True, False])
def test_lost_resume_discards_the_delta_turn_and_redoes_it_as_a_full_send(
    agy, make_client, registry, caplog, stream
):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    assert len(registry) == 1

    agy.lose_resume = True
    b = make_client()
    with caplog.at_level("WARNING"):
        if stream:
            output = _texts(_turn(b, [SYSTEM, U1, A1, U2]))
        else:
            completion = b.chat.completions.create(
                model=MODEL, messages=[SYSTEM, U1, A1, U2], hermes_session_id=SESSION
            )
            output = completion.choices[0].message.content

    first, lost, redo = agy.spawns
    assert lost["args"][lost["args"].index("--conversation") + 1] == first["conv"]
    # The resumed worker was never given the delta, so it produced nothing.
    lost["proc"].stdin.write.assert_not_called()
    assert "--conversation" not in redo["args"]
    assert "turn one" in agy.prompt_sent(2)  # the full history went out
    assert output == "reply 2.0"  # only the full send's output reached Hermes
    assert "reply 1" not in output
    assert any(
        r.levelname == "WARNING" and "resume lost" in r.getMessage() and first["conv"] in r.getMessage()
        for r in caplog.records
    )
    # The snapshot agy refused is gone; the redone turn published its own.
    assert len(registry) == 1
    assert all(snap.conversation_id == redo["conv"] for snap in registry._snapshots.values())
    assert not registry.in_use(first["conv"])


def test_lost_resume_after_an_idle_reap_also_falls_back(agy, make_client, caplog):
    a = make_client()
    _turn(a, [U1], session=None)
    a._reap_idle_worker(a._worker_proc)  # keeps the conversation for --conversation resume
    assert a._worker_conversation_id == agy.spawns[0]["conv"]

    agy.lose_resume = True
    with caplog.at_level("WARNING"):
        output = _texts(_turn(a, [U1, A1, U2], session=None))
    assert len(agy.spawns) == 3
    agy.spawns[1]["proc"].stdin.write.assert_not_called()
    assert "--conversation" not in agy.spawns[2]["args"]
    assert "turn one" in agy.prompt_sent(2)
    assert output == "reply 2.0"
    assert any("resume lost" in r.getMessage() for r in caplog.records)
    # The new conversation is the client's now; the next turn continues it.
    agy.lose_resume = False
    _turn(a, [U1, A1, U2, A2, U3], session=None)
    assert len(agy.spawns) == 3  # live worker took the delta


def test_unverifiable_resume_is_treated_like_a_lost_one(agy, make_client, registry, caplog):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    agy.silent_resume = True
    b = make_client()
    with caplog.at_level("WARNING"):
        output = _texts(_turn(b, [SYSTEM, U1, A1, U2]))
    assert len(agy.spawns) == 3
    agy.spawns[1]["proc"].stdin.write.assert_not_called()
    assert "--conversation" not in agy.spawns[2]["args"]
    assert output == "reply 2.0"
    assert any("resume unverified" in r.getMessage() for r in caplog.records)


def test_matching_conversation_id_is_not_disturbed_by_the_check(agy, make_client):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    b = make_client()
    output = _texts(_turn(b, [SYSTEM, U1, A1, U2]))
    assert len(agy.spawns) == 2
    assert output == "reply 1.0"
    agy.spawns[1]["proc"].stdin.write.assert_called()


# -- one in-flight turn per conversation id ---------------------------------------


def test_fork_during_an_in_flight_turn_is_a_full_send_and_release_allows_adoption(
    agy, make_client, registry
):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    conv = agy.spawns[0]["conv"]
    assert not registry.in_use(conv)

    main = make_client()
    in_flight = main.chat.completions.create(
        model=MODEL, messages=[SYSTEM, U1, A1, U2], stream=True, hermes_session_id=SESSION
    )
    assert registry.in_use(conv)  # adopted and being used

    review = make_client()
    _turn(review, [SYSTEM, U1, A1, {"role": "user", "content": "REVIEW"}])
    assert "--conversation" not in agy.spawns[2]["args"]
    assert agy.spawns[2]["rows_before"] == 0
    assert "turn one" in agy.prompt_sent(2)

    list(in_flight)  # the main turn completes, is snapshotted and releases
    assert not registry.in_use(conv)

    later = make_client()
    _turn(later, [SYSTEM, U1, A1, U2, A2, U3])
    assert agy.spawns[3]["args"][agy.spawns[3]["args"].index("--conversation") + 1] == conv
    assert agy.spawns[3]["rows_before"] == 2


def test_claim_is_released_when_the_turn_fails(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    conv = agy.spawns[0]["conv"]

    agy.fail_next = True
    b = make_client()
    with pytest.raises(RuntimeError):
        _turn(b, [SYSTEM, U1, A1, U2])
    assert not registry.in_use(conv)

    c = make_client()
    _turn(c, [SYSTEM, U1, A1, U2])
    assert "--conversation" in agy.spawns[2]["args"]  # adoptable again


def test_claim_is_released_when_the_stream_is_abandoned_and_closed(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    conv = agy.spawns[0]["conv"]
    b = make_client()
    stream = b.chat.completions.create(
        model=MODEL, messages=[SYSTEM, U1, A1, U2], stream=True, hermes_session_id=SESSION
    )
    assert registry.in_use(conv)
    stream.close()
    assert not registry.in_use(conv)


def test_claim_is_released_at_close(agy, make_client, registry):
    a = make_client()
    _turn(a, [SYSTEM, U1])
    a.close()
    conv = agy.spawns[0]["conv"]
    b = make_client()
    b.chat.completions.create(
        model=MODEL, messages=[SYSTEM, U1, A1, U2], stream=True, hermes_session_id=SESSION
    )
    assert registry.in_use(conv)
    b.close()
    assert not registry.in_use(conv)


def test_own_continuation_on_a_claimed_conversation_is_a_full_send(agy, make_client, registry):
    # Another client holds the id (a fork adopted it): this client's own delta
    # on that id would be a second writer, so it starts a fresh conversation.
    a = make_client()
    _turn(a, [SYSTEM, U1])
    conv = agy.spawns[0]["conv"]
    class Rival:
        pass

    rival = Rival()
    registry.release_owner(a)
    assert registry.claim(conv, rival)
    _turn(a, [SYSTEM, U1, A1, U2])
    assert len(agy.spawns) == 2
    assert "--conversation" not in agy.spawns[1]["args"]
    assert "turn one" in agy.prompt_sent(1)


# -- claim registry unit tests ------------------------------------------------------


def test_claim_is_exclusive_idempotent_and_releasable(tmp_path):
    reg = SessionRegistry(store_dir=tmp_path / "store")

    class Owner:
        pass

    a, b = Owner(), Owner()
    assert reg.claim("c1", a) and reg.claim("c1", a)
    assert not reg.claim("c1", b)
    assert reg.in_use("c1")
    reg.release_owner(b)  # not the holder: nothing happens
    assert reg.in_use("c1")
    reg.release_owner(a)
    assert not reg.in_use("c1")
    assert reg.claim("c1", b)
    assert not reg.claim("", a)
    reg.shutdown()


def test_an_owner_holds_one_claim_at_a_time(tmp_path):
    reg = SessionRegistry(store_dir=tmp_path / "store")

    class Owner:
        pass

    a = Owner()
    reg.claim("c1", a)
    reg.claim("c2", a)
    assert not reg.in_use("c1") and reg.in_use("c2")
    reg.shutdown()


def test_claim_dies_with_its_owner_and_after_the_hour(tmp_path, monkeypatch):
    import gc

    reg = SessionRegistry(store_dir=tmp_path / "store")

    class Owner:
        pass

    a = Owner()
    reg.claim("c1", a)
    del a
    gc.collect()
    assert not reg.in_use("c1")

    b = Owner()
    reg.claim("c2", b)
    monkeypatch.setattr(session_registry, "_CLAIM_MAX_SECONDS", -1.0)
    assert not reg.in_use("c2")
    reg.shutdown()


def test_lookup_skips_conversations_claimed_by_others_and_claims_for_its_owner(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store")

    class Owner:
        pass

    a, b = Owner(), Owner()
    _seed(reg, "k", 0, source=source_db, history=[U1], conv="c1")
    snap = reg.lookup("k", [U1, A1], owner=a)
    assert snap is not None and reg.in_use("c1")
    assert reg.lookup("k", [U1, A1], owner=b) is None  # held by a
    assert reg.lookup("k", [U1, A1]) is not None  # no owner: claims are not consulted
    assert reg.lookup("k", [U1, A1], owner=a) is not None  # idempotent for the holder
    reg.release_owner(a)
    assert reg.lookup("k", [U1, A1], owner=b) is not None
    reg.shutdown()


def test_drop_removes_the_snapshot_and_its_file(tmp_path, source_db):
    reg = SessionRegistry(store_dir=tmp_path / "store")
    sid = _seed(reg, "k", 0, source=source_db, history=[U1])
    path = reg._snapshots[sid].db_path
    reg.drop(None)
    reg.drop("unknown")
    assert len(reg) == 1
    reg.drop(sid)
    assert len(reg) == 0 and not path.exists()
    reg.shutdown()
