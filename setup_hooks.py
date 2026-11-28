"""Hermes setup, model picker, and auxiliary/vision hooks for the Antigravity provider.

Hermes asks external-process providers four questions outside inference:
- ``setup_status``: is the CLI installed and signed in, and what is the plan/quota detail?
- ``discover_models``: what base models and efforts should the picker show?
- ``resolve_aux_model``: what cheapest live model should handle auxiliary work?
- ``default_vision_model``: which live model accepts native image input?

All four are answered from the CLI itself without spending inference tokens:
``agy models`` for the catalog, and the OAuth token file plus the cached
``/usage`` read for sign-in status and quota detail.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install the Antigravity CLI (`agy`) and sign in once to use this provider."
LOGIN_HINT = "Run `agy` once in a terminal and finish the Google sign-in, then retry."

# setup_status is a UI read: never let the uncached keyring probe dominate it.
_AUTH_PROBE_DEADLINE = 8.0


def _helpers():
    try:
        from .process import is_authenticated, resolve_agy_command
    except ImportError:
        from process import is_authenticated, resolve_agy_command  # type: ignore[no-redef]
    return resolve_agy_command, is_authenticated


def _is_executable_command(command: Any, env: Optional[dict[str, str]] = None) -> bool:
    """Return True only when *command* resolves to a real executable binary."""
    if isinstance(command, (list, tuple)):
        command = command[0] if command else ""
    if not command or not isinstance(command, str):
        return False
    cmd = command.strip()
    if not cmd:
        return False
    if os.path.sep in cmd or (os.altsep and os.altsep in cmd):
        return os.path.isfile(cmd) and (os.name == "nt" or os.access(cmd, os.X_OK))
    search_path = env.get("PATH") if isinstance(env, dict) and "PATH" in env else None
    return bool(shutil.which(cmd, path=search_path))


def _bounded_authenticated(probe, deadline: float) -> bool:
    """Run ``is_authenticated()`` with a deadline, reporting False if it overruns.

    That probe is uncached and its documented ceiling is ~20 s (secret-tool 5 s,
    then up to three scripted interpreters at 5 s each). A per-request cost that
    size is invisible next to an agy spawn, but ``setup_status`` is a UI read
    where it looks like a hang. A timed-out probe falls through to the cached
    ``/usage`` snapshot, which needs the same sign-in and answers for it.
    """
    verdict: list[bool] = []

    def _run() -> None:
        try:
            verdict.append(bool(probe()))
        except Exception:
            verdict.append(False)

    worker = threading.Thread(target=_run, name="agy-auth-probe", daemon=True)
    worker.start()
    worker.join(max(0.1, float(deadline)))
    return bool(verdict[0]) if verdict else False


def _usage_snapshot(timeout: float, *, command: Any = None, env: Optional[dict[str, str]] = None):
    """The CLI's free `/usage` read (isolated home, cached), or None."""
    if command is not None or env is not None:
        # The cached usage reader uses the default agy binary and process env;
        # avoid serving a mismatched snapshot for a caller-overridden target.
        return None
    try:
        try:
            from .usage import fetch_subscription_usage
        except ImportError:
            from usage import fetch_subscription_usage  # type: ignore[no-redef]
        return fetch_subscription_usage(timeout=timeout)
    except Exception as exc:  # pragma: no cover - CLI/parse failures
        logger.debug("Antigravity usage probe failed: %s", exc)
        return None


def _quota_detail(snapshot) -> str:
    """One short line from the usage snapshot, e.g. `Gemini Models 5h: 62% left`."""
    try:
        fallback = ""
        for group in getattr(snapshot, "groups", []) or []:
            for bucket in getattr(group, "buckets", []) or []:
                window = str(getattr(bucket, "window", "")).strip()
                remaining = round(float(bucket.remaining_fraction) * 100)
                if "5h" in window.lower() or "5 hour" in window.lower():
                    return f"{group.name} 5h: {remaining}% left"
                if not fallback:
                    label = window or "quota"
                    fallback = f"{group.name} {label}: {remaining}% left"
        return fallback
    except Exception:
        return ""


def setup_status(command: Any = None, env: Optional[dict[str, str]] = None, timeout: float = 15.0) -> dict[str, Any]:
    """``{available, logged_in, plan, detail, login_command}`` from the CLI itself."""
    resolve_agy_command, is_authenticated = _helpers()
    explicit_command = command is not None
    resolved = command
    if not resolved:
        if isinstance(env, dict):
            for var in ("ANTIGRAVITY_COMMAND", "AGY_CLI_PATH", "ANTIGRAVITY_CLI_PATH"):
                val = str(env.get(var) or "").strip()
                if val:
                    resolved = val
                    break
        if not resolved:
            try:
                resolved = resolve_agy_command()
            except Exception:
                resolved = None
    if not _is_executable_command(resolved, env=env):
        return {
            "available": False,
            "logged_in": False,
            "plan": "",
            "detail": INSTALL_HINT,
            "login_command": None,
        }
    # is_authenticated() checks both a non-empty OAuth token file (>10 bytes)
    # and OS keyring backends (Linux Secret Service, macOS Keychain, Windows).
    try:
        logged_in = _bounded_authenticated(
            is_authenticated, min(_AUTH_PROBE_DEADLINE, float(timeout or _AUTH_PROBE_DEADLINE))
        )
    except Exception:
        logged_in = False
    snapshot = _usage_snapshot(
        timeout,
        command=resolved if explicit_command else None,
        env=env,
    )
    if snapshot is not None:
        logged_in = True
    detail = _quota_detail(snapshot) if snapshot is not None else ""
    if not logged_in:
        detail = LOGIN_HINT
    return {
        "available": True,
        "logged_in": logged_in,
        "plan": "Antigravity",
        "detail": detail,
        # The CLI signs in interactively; there is no non-interactive login verb.
        "login_command": None,
    }


def discover_models(timeout: float = 15.0) -> Optional[list[dict[str, str]]]:
    """The account's live picker as ``[{id, label, note}]``, or None.

    None covers both a failed probe and an unparseable empty catalog on purpose:
    Hermes' catalog cache treats a plain list as authoritative, and an empty
    parse is far likelier a transient ``agy models`` failure than an account
    that owns no models.
    """
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception as exc:
        logger.debug("Antigravity model discovery failed: %s", exc)
        return None
    return rows or None


def _effort_variant(row: dict[str, str]) -> str:
    """The row's cheapest callable id: its first effort variant, or the base id.

    Rows list efforts cheapest first. A base model that is only sold as effort
    variants (e.g. ``gemini-4-hydrogen``) rejects the bare id, so the bare id is
    correct only when the row has no variants at all.
    """
    base = row["id"]
    efforts = [e.strip() for e in (row.get("note") or "").split(",") if e.strip()]
    return f"{base}-{efforts[0]}" if efforts else base


def resolve_aux_model(timeout: float = 15.0) -> str:
    """Cheapest live model id for auxiliary work, or "" when unknown."""
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception:
        return ""
    by_id = {row["id"]: row for row in rows}
    for base in ("gemini-3.8-flash", "gemini-4-neon", "gemini-4-hydrogen"):
        row = by_id.get(base)
        if row is not None:
            return _effort_variant(row)
    for row in rows:
        if row["id"].startswith("gemini-4-"):
            return _effort_variant(row)
    return ""


def default_vision_model(timeout: float = 15.0) -> Optional[str]:
    """A model id that accepts image attachments natively, or None."""
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception:
        return None
    ids = [row["id"] for row in rows]
    for candidate in ("gemini-4-neon", "gemini-4-hydrogen", "gemini-4-carbon", "gemini-3.8-flash"):
        if candidate in ids:
            return candidate
    return next((model for model in ids if model.startswith("gemini")), None)
