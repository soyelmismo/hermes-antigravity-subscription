"""Isolate the suite from the developer's real Antigravity registry and env.

Without this, tests silently inherit whatever the local machine has configured
(a real registry with accounts, a persistent rotation mode, a pinned account),
so the same commit can pass on CI and fail on a configured machine — or vice
versa. Every test starts from: no registered accounts, rotation off.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_antigravity_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_ACCOUNTS_FILE", str(tmp_path / "accounts.json"))
    monkeypatch.setenv("ANTIGRAVITY_ACCOUNTS_DIR", str(tmp_path / "accounts"))
    monkeypatch.setenv("ANTIGRAVITY_ROTATION", "off")
    monkeypatch.delenv("ANTIGRAVITY_ACCOUNT", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_FAILOVER", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_SESSION_STICKINESS", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_QUOTA_IGNITION", raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_real_agy_model_probe(monkeypatch):
    """pick_account probes `agy models` per account; never spawn the real CLI from tests.

    An empty listing means "unknown" and fails open, which keeps rotation tests
    independent of the machine's accounts. Tests that need a listing patch it.
    """
    for name, mod in list(sys.modules.items()):
        if name.split(".")[-1] == "accounts" and hasattr(mod, "list_models_for_home"):
            monkeypatch.setattr(mod, "list_models_for_home", lambda home_dir, timeout=15.0: ())
    yield
