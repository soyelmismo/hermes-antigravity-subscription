"""Account registry and quota-based rotation for Antigravity accounts."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .process import resolve_agy_command
except ImportError:
    from process import resolve_agy_command

logger = logging.getLogger(__name__)

DEFAULT_ACCOUNTS_FILE = Path.home() / ".hermes" / "antigravity-accounts.json"
DEFAULT_ACCOUNTS_DIR = Path.home() / ".agy-accounts"

# Global in-memory cache for usage data: home_dir -> (timestamp, data)
_USAGE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_USAGE_CACHE_LOCK = threading.Lock()

# In-memory lease tracking to penalize busy accounts across concurrent turns
_LEASE_LOCK = threading.Lock()
_ACTIVE_LEASES: dict[str, int] = {}

# In-memory session pins: session_id -> {"label": str, "ts": float}
# Step 3 of the task brief: one account per session keeps the prompt prefix
# stable and Google's KV/prompt cache warm. In-memory only (process restart
# re-pins), guarded by a lock, entries expire after _SESSION_PIN_TTL.
_SESSION_PIN_LOCK = threading.Lock()
_SESSION_PINS: dict[str, dict[str, Any]] = {}
_SESSION_PIN_TTL = 6 * 3600.0

# In-memory quota-ignition bookkeeping: label -> monotonic-ish wall ts of the
# last fired ignition (Once per 5h window, see maybe_quota_ignition).
_IGNITION_LOCK = threading.Lock()
_IGNITION_FIRED: dict[str, float] = {}


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Cross-platform file lock for atomic registry updates."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as f:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                yield
            finally:
                with contextlib.suppress(Exception):
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                with contextlib.suppress(Exception):
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_accounts_file_path() -> Path:
    """Return path to registry file, respecting ANTIGRAVITY_ACCOUNTS_FILE env var."""
    env_path = os.environ.get("ANTIGRAVITY_ACCOUNTS_FILE", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_ACCOUNTS_FILE


def get_accounts_dir() -> Path:
    """Return path to accounts directory, respecting ANTIGRAVITY_ACCOUNTS_DIR env var."""
    env_dir = os.environ.get("ANTIGRAVITY_ACCOUNTS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_ACCOUNTS_DIR


def _default_registry() -> dict[str, Any]:
    return {
        "version": 1,
        "accounts": [],
        "active_account": None,
        "rotation_mode": "off",
    }


def load_accounts() -> dict[str, Any]:
    """Load account registry. Fails open on missing or corrupt files."""
    path = get_accounts_file_path()
    if not path.is_file():
        return _default_registry()

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, dict):
            return _default_registry()
        data.setdefault("version", 1)
        if not isinstance(data.get("accounts"), list):
            data["accounts"] = []
        if "active_account" not in data:
            data["active_account"] = None
        if "rotation_mode" not in data:
            data["rotation_mode"] = "off"
        return data
    except Exception as exc:
        logger.warning("Failed to read Antigravity accounts registry %s: %s (failing open)", path, exc)
        return _default_registry()


def save_accounts(data: dict[str, Any]) -> None:
    """Atomically save account registry with 0600 permissions under file lock."""
    path = get_accounts_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    lock_path = path.with_suffix(".lock")
    with _file_lock(lock_path):
        tmp_path = path.with_suffix(".tmp")
        try:
            content = json.dumps(data, indent=2)
            tmp_path.write_text(content, encoding="utf-8")
            try:
                tmp_path.chmod(0o600)
            except OSError:
                pass
            tmp_path.replace(path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        finally:
            if tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()


def sanitize_folder_name(name: str) -> str:
    """Sanitize a label or email for safe filesystem folder naming across OSes."""
    # Strip leading/trailing dots and dashes: the result is joined onto the
    # accounts directory, so a label such as '..' would address its parent.
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", name.strip()).strip(".-")
    return sanitized or "account"


def find_account_token_path(home_dir: str | Path) -> Path | None:
    """Locate a valid OAuth token file inside an account home directory."""
    base = Path(home_dir).expanduser().resolve() / ".gemini" / "antigravity-cli"
    for filename in ("jetski-standalone-oauth-token", "antigravity-oauth-token"):
        cand = base / filename
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
        except OSError:
            pass
    return None


def extract_email_from_token_file(token_path: str | Path) -> str | None:
    """Decode JWT id_token payload from token file and extract email.

    Does not log or expose the raw token content.
    """
    try:
        content = Path(token_path).read_text(encoding="utf-8")
        data = json.loads(content)
        id_token = data.get("id_token")
        if not id_token or not isinstance(id_token, str):
            return None
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
        email = payload.get("email")
        if email and isinstance(email, str) and email.strip():
            return email.strip()
    except Exception as exc:
        logger.debug("Failed to extract email from token file %s: %s", token_path, exc)
    return None


def check_account_eligibility(
    home_dir: str | Path,
    timeout: float = 30.0,
) -> tuple[bool, dict[str, Any] | None]:
    """Check account eligibility via `agy -p /usage --output-format json`.

    Returns (is_eligible, parsed_usage_or_None).
    """
    norm_home = str(Path(home_dir).expanduser().resolve())
    try:
        cmd = resolve_agy_command()
    except Exception:
        return False, None

    env = probe_env(norm_home)

    try:
        res = subprocess.run(
            [cmd, "-p", "/usage", "--output-format", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = f"{res.stdout}\n{res.stderr}".lower()
        if "eligibility check failed" in combined or "not eligible" in combined:
            return False, None
        if res.returncode != 0:
            return False, None

        parsed = _parse_usage_json(res.stdout)
        if parsed is not None:
            with _USAGE_CACHE_LOCK:
                _USAGE_CACHE[norm_home] = (time.monotonic(), parsed)
            return True, parsed

        try:
            raw_data = json.loads(res.stdout)
            if isinstance(raw_data, dict) and raw_data.get("status") == "SUCCESS":
                return True, None
        except Exception:
            pass

        return False, None
    except Exception as exc:
        logger.debug("Eligibility check error for %s: %s", norm_home, exc)
        return False, None


def add_account(
    label: str,
    home_dir: str,
    enabled: bool = True,
    eligible: bool = True,
    email: str | None = None,
) -> dict[str, Any]:
    """Add or update an account entry in the registry."""
    norm_label = label.strip()
    norm_home = str(Path(home_dir).expanduser().resolve())
    data = load_accounts()
    accounts = data.get("accounts", [])

    target = None
    for acc in accounts:
        if acc.get("label") == norm_label:
            target = acc
            break

    if target is not None:
        target["home_dir"] = norm_home
        target["enabled"] = bool(enabled)
        target["eligible"] = bool(eligible)
        if email:
            target["email"] = email
    else:
        target = {
            "label": norm_label,
            "home_dir": norm_home,
            "enabled": bool(enabled),
            "eligible": bool(eligible),
            "last_used": 0.0,
            "cooldown_until": 0.0,
        }
        if email:
            target["email"] = email
        accounts.append(target)

    data["accounts"] = accounts
    save_accounts(data)
    return target


def remove_account(label: str) -> bool:
    """Remove an account from registry. Preserves home directory on disk."""
    norm_label = label.strip()
    data = load_accounts()
    accounts = data.get("accounts", [])
    filtered = [acc for acc in accounts if acc.get("label") != norm_label]

    if len(filtered) == len(accounts):
        return False

    data["accounts"] = filtered
    if data.get("active_account") == norm_label:
        data["active_account"] = None
    save_accounts(data)
    return True


def list_accounts() -> list[dict[str, Any]]:
    """List all accounts in registry."""
    return list(load_accounts().get("accounts", []))


def get_active() -> str | None:
    """Get currently active account label, if set."""
    return load_accounts().get("active_account")


def set_active(label: str | None) -> bool:
    """Set or clear the active account label."""
    data = load_accounts()
    if label is None:
        data["active_account"] = None
        save_accounts(data)
        return True

    norm_label = label.strip()
    accounts = data.get("accounts", [])
    if any(acc.get("label") == norm_label for acc in accounts):
        data["active_account"] = norm_label
        save_accounts(data)
        return True
    return False


def set_cooldown(label: str, duration_seconds: float = 900.0, until: float | None = None) -> None:
    """Place an account on cooldown until a timestamp or for duration_seconds."""
    norm_label = label.strip()
    data = load_accounts()
    accounts = data.get("accounts", [])
    until_ts = until if until is not None else (time.time() + duration_seconds)
    found = False
    for acc in accounts:
        if acc.get("label") == norm_label:
            acc["cooldown_until"] = float(until_ts)
            found = True
            break
    if found:
        save_accounts(data)


def update_last_used(label: str) -> None:
    """Record account usage timestamp."""
    norm_label = label.strip()
    data = load_accounts()
    accounts = data.get("accounts", [])
    found = False
    for acc in accounts:
        if acc.get("label") == norm_label:
            acc["last_used"] = time.time()
            found = True
            break
    if found:
        save_accounts(data)


SERVING_KEY = "serving"
# Each request records the account it runs on; skip rewriting the registry when
# nothing changed and the record is still fresh.
_SERVING_REFRESH_SECONDS = 60.0
_SERVING_LOCK = threading.Lock()
_LAST_SERVING: tuple[str, str | None, float] | None = None


def record_serving_account(label: str | None, model: str | None = None) -> None:
    """Persist which account serves Antigravity requests right now.

    ``label`` is a registered account, or None/"" for the host default account
    (the real HOME login). Quota views (the TUI quota dock's probe, `/usage`) read
    it to meter the account actually in use instead of always the host default.
    Best-effort: a failed write never affects the request.
    """
    global _LAST_SERVING
    norm = (label or "").strip()
    now = time.time()
    with _SERVING_LOCK:
        last = _LAST_SERVING
        if last and last[0] == norm and last[1] == model and now - last[2] < _SERVING_REFRESH_SECONDS:
            return
        _LAST_SERVING = (norm, model, now)
    try:
        data = load_accounts()
        data[SERVING_KEY] = {"label": norm, "model": model, "at": now}
        save_accounts(data)
    except Exception as exc:
        logger.debug("record_serving_account failed: %s", exc)


def get_serving_account() -> dict[str, Any] | None:
    """The registry account last recorded as serving, or None for the host default."""
    try:
        data = load_accounts()
    except Exception:
        return None
    label = str((data.get(SERVING_KEY) or {}).get("label") or "")
    if not label:
        return None
    return next((a for a in data.get("accounts", []) if a.get("label") == label), None)


def parse_reset_time(reset_time_str: str | None) -> float | None:
    """Parse ISO 8601 reset_time string to Unix epoch timestamp."""
    if not reset_time_str or not isinstance(reset_time_str, str):
        return None
    try:
        clean = reset_time_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            # agy reports UTC; a bare stamp must not be read as local time.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _parse_usage_json(raw_json: str) -> dict[str, Any] | None:
    """Parse `agy -p /usage --output-format json` payload into group buckets."""
    try:
        data = json.loads(raw_json)
        cmd_data = data.get("command", {}).get("data", {})
        groups = cmd_data.get("groups", [])
        if not groups and isinstance(data.get("groups"), list):
            groups = data["groups"]

        parsed: dict[str, Any] = {
            "gemini": {"5h": {"remaining_fraction": 1.0, "reset_time": None}, "weekly": {"remaining_fraction": 1.0, "reset_time": None}},
            "claude_gpt": {"5h": {"remaining_fraction": 1.0, "reset_time": None}, "weekly": {"remaining_fraction": 1.0, "reset_time": None}},
        }

        for grp in groups:
            grp_name = grp.get("name", "").lower()
            if "gemini" in grp_name:
                target = parsed["gemini"]
            elif any(k in grp_name for k in ("claude", "gpt", "3p")):
                target = parsed["claude_gpt"]
            else:
                continue

            for bucket in grp.get("buckets", []):
                bid = bucket.get("id", "").lower()
                window = bucket.get("window", "").lower()
                rem = bucket.get("remaining_fraction")
                reset_t = bucket.get("reset_time")
                try:
                    frac = float(rem) if rem is not None else 1.0
                except (ValueError, TypeError):
                    frac = 1.0

                if window == "5h" or "5h" in bid:
                    target["5h"] = {"remaining_fraction": frac, "reset_time": reset_t}
                elif window == "weekly" or "weekly" in bid:
                    target["weekly"] = {"remaining_fraction": frac, "reset_time": reset_t}

        return parsed
    except Exception as exc:
        logger.debug("Failed to parse Antigravity usage payload: %s", exc)
        return None


_NO_BROWSER_LOCK = threading.Lock()
_NO_BROWSER_DIR: str | None = None


def is_browser_block_enabled() -> bool:
    """Whether probe subprocesses are blocked from launching a browser. Default: on.

    Disable with ANTIGRAVITY_BLOCK_BROWSER=0 (only useful for interactive
    sign-in, where a browser window is the whole point).
    """
    val = os.environ.get("ANTIGRAVITY_BLOCK_BROWSER", "").strip().lower()
    return val not in ("0", "false", "no", "off")


def _no_browser_shim_dir() -> str:
    """Directory holding `open`/`xdg-open` shims that refuse to launch a browser.

    `agy` opens the system browser when it decides a session needs re-login.
    A quota probe (list, `/usage`, status bar, desktop) must never do that:
    unattended probes would pop a Chrome tab the user never asked for. We
    prepend this directory to PATH so those helpers fail instead of launching.
    """
    global _NO_BROWSER_DIR
    with _NO_BROWSER_LOCK:
        if _NO_BROWSER_DIR and os.path.isdir(_NO_BROWSER_DIR):
            return _NO_BROWSER_DIR
        shim_dir = os.path.join(tempfile.gettempdir(), "agy-nobrowser")
        os.makedirs(shim_dir, exist_ok=True)
        for name in ("open", "xdg-open"):
            path = os.path.join(shim_dir, name)
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(
                        '#!/bin/sh\n'
                        'echo "blocked browser launch (agy probe): $*" >&2\n'
                        "exit 1\n"
                    )
                os.chmod(path, 0o755)
            except OSError:
                return ""
        _NO_BROWSER_DIR = shim_dir
        return shim_dir


def apply_browser_block(env: dict[str, str]) -> dict[str, str]:
    """Block a subprocess env from launching a browser (BROWSER + `open` shim)."""
    if not is_browser_block_enabled():
        return env
    env["BROWSER"] = "/usr/bin/false"
    shim_dir = _no_browser_shim_dir()
    if shim_dir:
        env["PATH"] = shim_dir + os.pathsep + env.get("PATH", "")
    return env


def probe_env(home_dir: str) -> dict[str, str]:
    """Environment for a read-only `agy` probe: isolated HOME, no browser launch."""
    env = os.environ.copy()
    env["HOME"] = str(Path(home_dir).expanduser().resolve())
    env.pop("ANTIGRAVITY_CONFIG_DIR", None)
    return apply_browser_block(env)


def fetch_usage_for_home(
    home_dir: str,
    timeout: float = 30.0,
    cached: bool = True,
    max_cache_age: float = 60.0,
) -> dict[str, Any] | None:
    """Query quota for home_dir using `agy -p /usage --output-format json`."""
    norm_home = str(Path(home_dir).expanduser().resolve())
    now = time.monotonic()

    if cached:
        with _USAGE_CACHE_LOCK:
            if norm_home in _USAGE_CACHE:
                ts, cached_data = _USAGE_CACHE[norm_home]
                if now - ts < max_cache_age:
                    return cached_data

    try:
        cmd = resolve_agy_command()
    except Exception:
        return None

    env = probe_env(norm_home)

    try:
        res = subprocess.run(
            [cmd, "-p", "/usage", "--output-format", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res.returncode != 0:
            return None
        parsed = _parse_usage_json(res.stdout)
        if parsed is not None:
            with _USAGE_CACHE_LOCK:
                _USAGE_CACHE[norm_home] = (now, parsed)
        return parsed
    except Exception as exc:
        logger.debug("Usage check failed for %s: %s", norm_home, exc)
        return None


VALID_ROTATION_MODES = ("off", "quota", "round_robin", "fixed")

# A quota probe that fails must never look like a full tank: the account is
# scored half-full and then damped, so any account with a known healthy quota
# wins the pick and an unreadable account is only used when nothing else is left.
_UNKNOWN_USAGE_FRACTION = 0.5
_UNKNOWN_USAGE_PENALTY = 0.5

# Opt-in quota ignition: start the 5-hour usage window on an idle account so the
# full budget is available when the user actually needs it. Disabled by default —
# see `maybe_quota_ignition` for the triggers and safety rules.
_QUOTA_IGNITION_IDLE_FRACTION = 0.999
_QUOTA_IGNITION_TIMEOUT = 30.0
_QUOTA_IGNITION_5H_WINDOW_HOURS = 5.0 * 3600.0


def _registry_rotation_mode() -> str | None:
    """Return rotation_mode stored in the registry, validating known modes."""
    stored = load_accounts().get("rotation_mode")
    if isinstance(stored, str):
        stored = stored.strip().lower()
        if stored in VALID_ROTATION_MODES:
            return stored
        logger.warning("Unknown rotation_mode %r in registry; ignoring it.", stored)
    return None


def get_rotation_mode() -> str:
    """Return the active rotation mode.

    Precedence: ANTIGRAVITY_ROTATION env (session-level override), then the
    persistent ``rotation_mode`` registry field, then ``off``.
    """
    env_mode = os.environ.get("ANTIGRAVITY_ROTATION", "").strip().lower()
    if env_mode:
        if env_mode in VALID_ROTATION_MODES:
            return env_mode
        logger.warning("Unknown ANTIGRAVITY_ROTATION %r; falling back to registry.", env_mode)
    return _registry_rotation_mode() or "off"


def set_rotation_mode(mode: str) -> bool:
    """Persist rotation mode to the registry. Returns False on an unknown mode."""
    norm = (mode or "").strip().lower()
    if norm not in VALID_ROTATION_MODES:
        logger.warning("Invalid rotation mode %r (valid: %s).", mode, ", ".join(VALID_ROTATION_MODES))
        return False
    data = load_accounts()
    data["rotation_mode"] = norm
    save_accounts(data)
    return True


def is_quota_ignition_enabled() -> bool:
    """Return True when quota ignition is enabled via env or registry (opt-in)."""
    env_val = os.environ.get("ANTIGRAVITY_QUOTA_IGNITION", "").strip().lower()
    if env_val:
        return env_val not in ("0", "false", "no", "off")
    data = load_accounts()
    return bool(data.get("quota_ignition", False))


def acquire_lease(label: str) -> None:
    """Take a turn lease on an account without entering a context manager."""
    norm = label.strip()
    with _LEASE_LOCK:
        _ACTIVE_LEASES[norm] = _ACTIVE_LEASES.get(norm, 0) + 1


def release_lease(label: str) -> None:
    """Drop one turn lease; releasing an unleased account is a no-op."""
    norm = label.strip()
    with _LEASE_LOCK:
        remaining = _ACTIVE_LEASES.get(norm, 0) - 1
        if remaining > 0:
            _ACTIVE_LEASES[norm] = remaining
        else:
            _ACTIVE_LEASES.pop(norm, None)


@contextlib.contextmanager
def lease_account(label: str):
    """Thread-safe context manager tracking in-flight turn leases per account."""
    acquire_lease(label)
    try:
        yield
    finally:
        release_lease(label)


def get_lease_count(label: str) -> int:
    """Return active lease count for an account label."""
    with _LEASE_LOCK:
        return _ACTIVE_LEASES.get(label.strip(), 0)


def calculate_score(
    f_5h: float,
    f_weekly: float,
    in_cooldown: bool,
    lease_count: int = 0,
    eligible: bool = True,
    reset_5h_time: str | None = None,
    reset_week_time: str | None = None,
) -> float:
    """Calculate the selection score with hard gate at 0%, cooldown, or ineligible.

    Concurrency-Dampened DOCI shape (credit: prmartinow's CD-DOCI rotation):

      base = f_5h * f_weekly^2
      U_5h = (5.0 / (t_5h_hours + 0.5))^0.6      -- 5h replenishment urgency
      U_w  = 7.0 / (t_week_days + 0.5)           -- weekly replenishment urgency
      score = base * U_5h * U_w / (1 + leases)

    A window about to reset scores higher: burning the remainder now instead of
    letting it expire is the opportunity-cheap choice. Unknown/missing reset
    times fall back to the full window length (5 h / 7 days), i.e. the minimum
    urgency factor. Hard gates return 0.0 exactly.
    """
    if not eligible or in_cooldown or f_5h <= 0.0 or f_weekly <= 0.0:
        return 0.0

    now = time.time()
    t_5h_hours = 5.0
    reset_5h_ts = parse_reset_time(reset_5h_time)
    if reset_5h_ts is not None:
        t_5h_hours = max(0.001, (reset_5h_ts - now) / 3600.0)

    t_week_days = 7.0
    reset_week_ts = parse_reset_time(reset_week_time)
    if reset_week_ts is not None:
        t_week_days = max(0.001, (reset_week_ts - now) / 86400.0)

    u_5h = (5.0 / (t_5h_hours + 0.5)) ** 0.6
    u_w = 7.0 / (t_week_days + 0.5)
    raw_score = f_5h * (f_weekly ** 2) * u_5h * u_w
    # Penalty for active leases ensures concurrent turns distribute evenly
    return raw_score / (1.0 + lease_count)


def _quota_ignition_pinned(label: str) -> bool:
    """Whether an ignition was already fired for this label near this 5h window."""
    with _IGNITION_LOCK:
        ts = _IGNITION_FIRED.get(label)
    if ts is None:
        return False
    consumed = time.time() - ts
    if consumed >= _QUOTA_IGNITION_5H_WINDOW_HOURS:
        # Window must have rotated; forget the old stamp so the next idle
        # stretch can fire again.
        with _IGNITION_LOCK:
            if _IGNITION_FIRED.get(label) == ts:
                _IGNITION_FIRED.pop(label, None)
        return False
    return True


def maybe_quota_ignition(
    account: dict[str, Any],
    usage: dict[str, Any] | None = None,
    model_group: str = "gemini",
) -> bool:
    """Opt-in quota ignition: start an idle account's 5h window before use.

    Disabled by default (env ANTIGRAVITY_QUOTA_IGNITION=1 or registry
    ``quota_ignition: true``). Fires `HOME=<home> agy -p "Say: Ready"` ONCE per
    (account, 5h window) when the account sits idle (5h remaining >= 0.999) —
    quota is otherwise lost when the window resets. Never blocks the main
    request beyond that single bounded call, never runs for disabled or
    ineligible accounts, and never touches authentication state (a /usage read
    is read-only; the ignition prompt is the same subprocess shape the
    eligibility check already uses). Returns True when an ignition was fired.
    """
    label = str(account.get("label", "")).strip()
    if not label:
        return False
    if not account.get("enabled", True):
        return False
    if not account.get("eligible", True):
        return False
    if not is_quota_ignition_enabled():
        return False
    if _quota_ignition_pinned(label):
        return False

    group = usage.get(model_group) if usage and isinstance(usage, dict) else None
    if not isinstance(group, dict):
        return False
    f_5h = group.get("5h", {}).get("remaining_fraction", 0.0)
    try:
        f_5h = float(f_5h)
    except (TypeError, ValueError):
        return False
    if f_5h < _QUOTA_IGNITION_IDLE_FRACTION:
        return False

    home_dir = str(account.get("home_dir", "") or "").strip()
    if not home_dir or not Path(home_dir).is_dir():
        return False

    try:
        cmd = resolve_agy_command()
    except Exception:
        return False

    env = probe_env(home_dir)

    try:
        res = subprocess.run(
            [cmd, "-p", "Say: Ready"],
            env=env,
            capture_output=True,
            text=True,
            timeout=_QUOTA_IGNITION_TIMEOUT,
            check=False,
        )
    except Exception:
        return False
    if res.returncode != 0:
        logger.warning(
            "quota ignition for %s exited rc=%d (ignoring)", label, res.returncode
        )
        return False

    with _IGNITION_LOCK:
        _IGNITION_FIRED[label] = time.time()
    logger.info("quota ignition fired for account %s (5h window started)", label)
    return True


def is_quota_error(error: Exception | str) -> bool:
    """Detect quota exhaustion across error types, messages, and codes."""
    msg = str(error).lower()
    return any(
        term in msg
        for term in (
            "resource_exhausted",
            "individual quota reached",
            "quota exceeded",
            "quota limit",
            "429",
            # "rate limit" is deliberately absent: transient throttling is not
            # quota exhaustion and must not trigger an account swap.
        )
    )


def _is_session_stickiness_enabled() -> bool:
    env_val = os.environ.get("ANTIGRAVITY_SESSION_STICKINESS", "").strip().lower()
    if env_val:
        return env_val not in ("0", "false", "no", "off")
    return True


def _get_session_pin(session_id: str) -> str | None:
    """Return the label pinned for this session, pruning stale pins as we go."""
    now = time.time()
    with _SESSION_PIN_LOCK:
        # Cheap opportunistic sweep so pins don't accumulate across long sessions.
        for sid in [s for s, p in _SESSION_PINS.items() if now - p.get("ts", 0.0) > _SESSION_PIN_TTL]:
            _SESSION_PINS.pop(sid, None)
        pin = _SESSION_PINS.get(session_id)
        if pin is not None and now - pin.get("ts", 0.0) <= _SESSION_PIN_TTL:
            return str(pin.get("label") or "")
    return None


def _set_session_pin(session_id: str, label: str) -> None:
    with _SESSION_PIN_LOCK:
        _SESSION_PINS[session_id] = {"label": label, "ts": time.time()}


def clear_session_pin(session_id: str) -> None:
    with _SESSION_PIN_LOCK:
        _SESSION_PINS.pop(session_id, None)


def _account_passes_gates(acc: dict[str, Any], now: float) -> bool:
    """Enabled + eligible + not on cooldown."""
    if not acc.get("enabled", True):
        return False
    if not acc.get("eligible", True):
        return False
    if float(acc.get("cooldown_until", 0.0) or 0.0) > now:
        return False
    return True


def is_failover_enabled() -> bool:
    """Whether a quota error may swap to another account. Default: on.

    Deliberately independent of the rotation mode: even with rotation 'off'
    (host default account) a quota-exhausted turn should move to a healthy
    registered account instead of letting Hermes degrade to another model.
    Disable with ANTIGRAVITY_FAILOVER=0.
    """
    val = os.environ.get("ANTIGRAVITY_FAILOVER", "").strip().lower()
    return val not in ("0", "false", "no", "off")


def has_registered_accounts() -> bool:
    """Whether any account is registered. With none, nothing can be failed over to."""
    try:
        data = load_accounts()
    except Exception:
        return False
    return bool(data.get("accounts"))


def _account_has_quota(acc: dict[str, Any], group_key: str) -> bool:
    """Whether both windows of the model group still show remaining quota.

    A session pin must not survive quota exhaustion: pinning an account that is
    at 0% keeps sending turns into a 429 until the window resets. Unknown usage
    is treated as 'has quota' so a probe failure never drops a good pin.
    """
    try:
        usage = fetch_usage_for_home(acc.get("home_dir", ""), cached=True)
    except Exception:
        return True
    group = usage.get(group_key) if isinstance(usage, dict) else None
    if not isinstance(group, dict):
        return True
    try:
        f_5h = float(group.get("5h", {}).get("remaining_fraction", 1.0))
        f_weekly = float(group.get("weekly", {}).get("remaining_fraction", 1.0))
    except (TypeError, ValueError):
        return True
    return f_5h > 0.0 and f_weekly > 0.0


_MODELS_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_MODELS_CACHE_LOCK = threading.Lock()
# Model access changes rarely, and every probe may refresh the account's token
# (on macOS that refresh pops the "keychain cannot be found" dialog), so probe seldom.
_MODELS_CACHE_TTL = 6 * 3600.0
# A failed probe is remembered briefly so a broken account costs one timeout, not one per request.
_MODELS_FAILURE_TTL = 300.0


def list_models_for_home(home_dir: str | None, timeout: float = 15.0) -> tuple[str, ...]:
    """Raw `agy models` ids one account can use (cached); empty when unknown.

    ``home_dir=None`` lists the host default account (the real HOME). Model access
    is per Google account: one account may list Claude 5.5 while another only has
    4.6, and agy exits on a ``--model`` its account does not list.
    """
    key = str(home_dir or "")
    now = time.time()
    with _MODELS_CACHE_LOCK:
        hit = _MODELS_CACHE.get(key)
        if hit and now - hit[0] < (_MODELS_CACHE_TTL if hit[1] else _MODELS_FAILURE_TTL):
            return hit[1]
    models: tuple[str, ...] = ()
    try:
        env = probe_env(home_dir) if home_dir else apply_browser_block(dict(os.environ))
        res = subprocess.run(
            [resolve_agy_command(), "models"], env=env, capture_output=True, text=True, timeout=timeout,
        )
        models = tuple(dict.fromkeys(  # agy's own order, deduplicated
            line.split()[0] for line in (res.stdout or "").splitlines()
            if line.strip() and "fetching" not in line.lower()
        ))
    except Exception as exc:
        logger.debug("agy models probe failed for %s: %s", home_dir or "host default", exc)
    with _MODELS_CACHE_LOCK:  # empty = unknown; callers fail open on it
        _MODELS_CACHE[key] = (now, models)
    return models


def account_supports_model(acc: dict[str, Any], model: str | None) -> bool:
    """Whether the account lists ``model`` (exact id or an effort variant). Unknown: True."""
    if not model:
        return True
    # Runs on the request path: bound the probe tighter than the picker's listing.
    models = list_models_for_home(acc.get("home_dir") or None, timeout=8.0)
    if not models:
        return True  # probe failed: fail open rather than strand the request
    m = model.lower()
    return any(x.lower() == m or x.lower().startswith(m + "-") for x in models)


def pick_account(
    model: str | None = None,
    exclude_labels: set[str] | list[str] | None = None,
    session_id: str | None = None,
    force_mode: str | None = None,
) -> dict[str, Any] | None:
    """Pick the best available Antigravity account based on configured rotation mode.

    Mode precedence: env ANTIGRAVITY_ROTATION, else the registry's persistent
    ``rotation_mode``, else 'off'.

    ``force_mode`` overrides both for one call. Quota failover uses
    ``force_mode='quota'`` so a quota error can still swap accounts while
    rotation is 'off'.

    'fixed' pins the account chosen by `set_active` (registry's
    ``active_account``), or ANTIGRAVITY_ACCOUNT env (per-session override of the
    pin). Fail-open: an unavailable pin account falls back to host default
    (None).

    'quota' / 'round_robin' honor per-session stickiness (see session_id):
    once a session is pinned to an account it keeps that account (if it still
    passes gates) so the prompt prefix and Google's prompt cache stay warm.

    Fails open by returning None on any registry missing/corrupt or mode 'off'.
    """
    try:
        mode = (force_mode or "").strip().lower() or get_rotation_mode()
        if mode not in VALID_ROTATION_MODES:
            mode = "off"
        if mode == "off":
            return None

        data = load_accounts()
        accounts = data.get("accounts", [])
        if not accounts:
            return None
        by_label: dict[str, dict[str, Any]] = {
            str(acc.get("label", "") or ""): acc for acc in accounts
        }

        excluded = set(exclude_labels or [])
        # An account without the model makes agy exit on --model (BrokenPipe, then
        # fallback provider). Skip it; with none left the request goes to the host
        # default account, which may have the model.
        excluded.update(
            str(acc.get("label", "") or "") for acc in accounts
            if not account_supports_model(acc, model)
        )
        if session_id and not _is_session_stickiness_enabled():
            session_id = None  # stickiness disabled for this call

        if mode == "fixed":
            # Per-session env pin (ANTIGRAVITY_ACCOUNT) wins over the registry pin.
            pin_label = ""
            env_pin = os.environ.get("ANTIGRAVITY_ACCOUNT", "").strip()
            if env_pin:
                pin_label = env_pin
            else:
                pin_label = str(data.get("active_account") or "")
                if not pin_label:
                    return None  # fixed mode without a pin -> host default
            acc = by_label.get(pin_label)
            if acc is None or acc.get("label") in excluded or not _account_passes_gates(acc, time.time()):
                what = "env pin" if env_pin else "active_account"
                logger.warning(
                    "rotation mode 'fixed': %s '%s' unavailable (missing, excluded, disabled, "
                    "ineligible or on cooldown); failing open to host default.",
                    what, pin_label,
                )
                return None
            return acc

        # Mode 'quota' / 'round_robin' + session stickiness
        m_lower = (model or "").lower()
        group_key = "claude_gpt" if ("claude" in m_lower or "gpt" in m_lower) else "gemini"
        if session_id:
            pinned_label = _get_session_pin(session_id)
            if pinned_label:
                pinned_acc = by_label.get(pinned_label)
                if (
                    pinned_acc is not None
                    and pinned_acc.get("label") not in excluded
                    and _account_passes_gates(pinned_acc, time.time())
                    and (mode != "quota" or _account_has_quota(pinned_acc, group_key))
                ):
                    return pinned_acc
                # Pin went stale (removed / disabled / ineligible / cooldown /
                # exhausted): reselect below and re-pin.

        now = time.time()

        if mode == "round_robin":
            eligible = [
                acc for acc in accounts
                if _account_passes_gates(acc, now) and acc.get("label") not in excluded
            ]
            if not eligible:
                return None
            # Sort by active leases first (idle accounts preferred), then least recently used
            eligible.sort(key=lambda a: (get_lease_count(a["label"]), a.get("last_used", 0.0)))
            chosen = eligible[0]
        else:
            # Default mode: 'quota'
            scored: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []
            for acc in accounts:
                lbl = str(acc.get("label", "") or "")
                if not lbl or lbl in excluded or not _account_passes_gates(acc, now):
                    continue
                f_5h = _UNKNOWN_USAGE_FRACTION
                f_weekly = _UNKNOWN_USAGE_FRACTION
                reset_5h: str | None = None
                reset_week: str | None = None
                quota_known = False
                usage = fetch_usage_for_home(acc.get("home_dir", ""), cached=True)
                group = usage.get(group_key) if usage and isinstance(usage, dict) else None
                if isinstance(group, dict):
                    b5 = group.get("5h", {}) if isinstance(group.get("5h", {}), dict) else {}
                    bwk = group.get("weekly", {}) if isinstance(group.get("weekly", {}), dict) else {}
                    try:
                        raw_5h = b5.get("remaining_fraction")
                        raw_week = bwk.get("remaining_fraction")
                        if raw_5h is not None and raw_week is not None:
                            f_5h = float(raw_5h)
                            f_weekly = float(raw_week)
                            reset_5h = b5.get("reset_time") or None
                            reset_week = bwk.get("reset_time") or None
                            quota_known = True
                    except (TypeError, ValueError):
                        # A hand-edited or unexpected payload is 'unknown', not fatal.
                        quota_known = False

                score = calculate_score(
                    f_5h=f_5h,
                    f_weekly=f_weekly,
                    in_cooldown=False,
                    lease_count=get_lease_count(lbl),
                    eligible=acc.get("eligible", True),
                    reset_5h_time=reset_5h,
                    reset_week_time=reset_week,
                )
                if score > 0.0:
                    if not quota_known:
                        score *= _UNKNOWN_USAGE_PENALTY
                    scored.append((score, acc, usage))

            if not scored:
                return None

            # Highest score first
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = scored[0][1]
            # Opt-in quota ignition: if the winner's 5h window is idle, spend ONE
            # bounded subprocess call to start it (else that quota expires when
            # the window resets). Best-effort — failure never blocks the request.
            maybe_quota_ignition(chosen, usage=scored[0][2], model_group=group_key)

        if session_id:
            # Keep this session on the chosen account until it becomes unusable.
            _set_session_pin(session_id, str(chosen.get("label", "")))
        return chosen
    except Exception as exc:
        logger.warning("pick_account failed open: %s", exc)
        return None
