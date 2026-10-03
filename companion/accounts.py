"""Account registry and quota helpers for the Antigravity companion CLI.

Self-contained on purpose: this plugin installs separately from the
``antigravity-subscription-directsdk`` provider, so it reads the same on-disk
contract (the account registry JSON plus one isolated HOME per account) instead
of importing the provider package.

Contract read here:
  * registry file: ``~/.hermes/antigravity-accounts.json`` (env ``ANTIGRAVITY_ACCOUNTS_FILE``)
  * accounts dir:  ``~/.agy-accounts/<label>/`` (env ``ANTIGRAVITY_ACCOUNTS_DIR``)
  * token file:    ``<home>/.gemini/antigravity-cli/antigravity-oauth-token``
  * quota probe:   ``HOME=<home> agy -p "/usage" --output-format json`` (num_turns=0, no token cost)
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN_FILENAMES = ("jetski-standalone-oauth-token", "antigravity-oauth-token")
QUOTA_TIMEOUT_S = 45
HOST_LABEL = "host"
BINARY_ENV_VARS = ("ANTIGRAVITY_COMMAND", "AGY_CLI_PATH", "ANTIGRAVITY_CLI_PATH")
ROTATION_MODES = ("off", "quota", "round_robin", "fixed")


def accounts_dir() -> Path:
    raw = os.environ.get("ANTIGRAVITY_ACCOUNTS_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".agy-accounts"


def registry_file() -> Path:
    raw = os.environ.get("ANTIGRAVITY_ACCOUNTS_FILE", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".hermes" / "antigravity-accounts.json"


def agy_command() -> str:
    for var in BINARY_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return "agy"


def token_path_for(home: Path | str) -> Path | None:
    base = Path(home).expanduser()
    for name in TOKEN_FILENAMES:
        candidate = base / ".gemini" / "antigravity-cli" / name
        if candidate.is_file():
            return candidate
    return None


def email_from_token(path: Path | None) -> str:
    """Return the account email from the id_token JWT payload. Never returns or prints the token."""
    if path is None:
        return ""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return ""
    token = data.get("id_token") or (data.get("token") or {}).get("id_token") or ""
    parts = str(token).split(".")
    if len(parts) < 2:
        return ""
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return ""
    return str(claims.get("email") or "")


def load_registry() -> dict[str, Any]:
    try:
        data = json.loads(registry_file().read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def registry_accounts() -> list[dict[str, Any]]:
    entries = load_registry().get("accounts") or []
    return [entry for entry in entries if isinstance(entry, dict)]


def list_accounts(include_scan: bool = True) -> list[dict[str, Any]]:
    """Host account first, then registry entries, then any extra account dirs on disk."""
    seen_homes: set[str] = set()
    accounts: list[dict[str, Any]] = [
        {
            "label": HOST_LABEL,
            "home": Path.home(),
            "email": "",
            "eligible": None,
            "registered": False,
            "host": True,
        }
    ]
    seen_homes.add(str(Path.home().resolve()))

    for entry in registry_accounts():
        home_raw = str(entry.get("home_dir") or "").strip()
        if not home_raw:
            continue
        home = Path(home_raw).expanduser()
        key = str(home.resolve()) if home.exists() else str(home)
        if key in seen_homes:
            continue
        seen_homes.add(key)
        accounts.append(
            {
                "label": str(entry.get("label") or home.name),
                "home": home,
                "email": str(entry.get("email") or "") or email_from_token(token_path_for(home)),
                "eligible": entry.get("eligible"),
                "registered": True,
                "host": False,
            }
        )

    if include_scan and accounts_dir().is_dir():
        for child in sorted(accounts_dir().iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            key = str(child.resolve())
            if key in seen_homes:
                continue
            seen_homes.add(key)
            accounts.append(
                {
                    "label": child.name,
                    "home": child,
                    "email": email_from_token(token_path_for(child)),
                    "eligible": None,
                    "registered": False,
                    "host": False,
                }
            )
    return accounts


def resolve_account(label: str) -> dict[str, Any] | None:
    wanted = (label or "").strip()
    if not wanted:
        return None
    for account in list_accounts():
        if account["label"] == wanted or account["home"].name == wanted:
            return account
    return None


def probe(home: Path | str) -> tuple[bool | None, dict[str, float | None], str, dict[str, float]]:
    """(eligible, windows, note, reset_times). eligible is None when it cannot be determined."""
    env = os.environ.copy()
    env["HOME"] = str(Path(home).expanduser())
    env.pop("ANTIGRAVITY_CONFIG_DIR", None)
    try:
        result = subprocess.run(
            [agy_command(), "-p", "/usage", "--output-format", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=QUOTA_TIMEOUT_S,
        )
    except FileNotFoundError:
        return None, {}, "agy not found", {}
    except subprocess.TimeoutExpired:
        return None, {}, "timeout", {}

    stdout = (result.stdout or "").strip()
    blob = f"{stdout}\n{result.stderr or ''}".lower()
    if "not eligible" in blob:
        return False, {}, "not eligible for Antigravity", {}
    if "sign in" in blob:
        return None, {}, "not signed in", {}
    if not stdout:
        lines = (result.stderr or "").strip().splitlines()
        return None, {}, "", (lines[0][:60] if lines else f"exit {result.returncode}")

    try:
        data = json.loads(stdout)
    except Exception:
        return None, {}, "", stdout.splitlines()[0][:60]

    groups = ((data.get("command") or {}).get("data") or {}).get("groups") or []
    windows: dict[str, float | None] = {}
    resets: dict[str, float] = {}
    for group in groups:
        key = "gemini" if "gemini" in str(group.get("name", "")).lower() else "claude_gpt"
        for bucket in group.get("buckets") or []:
            window = str(bucket.get("window") or "").lower()
            fraction = bucket.get("remaining_fraction")
            windows[f"{key}_{window}"] = float(fraction) if isinstance(fraction, (int, float)) else None
            reset = parse_reset_time(bucket.get("reset_time"))
            if reset is not None:
                resets[f"{key}_{window}"] = reset
    return True, windows, "", resets



def run_agy(home: Path | str, args: list[str]) -> int:
    """Run the official agy binary with HOME pointed at one account. Returns its exit code."""
    env = os.environ.copy()
    env["HOME"] = str(Path(home).expanduser())
    env.pop("ANTIGRAVITY_CONFIG_DIR", None)
    try:
        return subprocess.call([agy_command(), *args], env=env)
    except FileNotFoundError:
        print(f"agy not found on PATH (looked for '{agy_command()}')")
        return 127


def find_account_dir(label: str) -> Path | None:
    """Best-effort location of a stored account dir matching a label (no scanning of live probes)."""
    wanted = (label or "").strip()
    if not wanted or wanted == HOST_LABEL:
        return None
    for entry in registry_accounts():
        current = (str(entry.get("label") or "").strip() or Path(str(entry.get("home_dir") or "")).name).strip()
        if current == wanted:
            home = Path(str(entry.get("home_dir") or "")).expanduser()
            return home if str(home) else None
    root = accounts_dir()
    if root.is_dir():
        child = root / wanted
        if child.is_dir():
            return child
    return None


def pct(value: float | None) -> str:
    return f"{round(value * 100)}%" if isinstance(value, (int, float)) else "-"


def window_pair(windows: dict[str, float | None], key: str) -> str:
    five, weekly = windows.get(f"{key}_5h"), windows.get(f"{key}_weekly")
    if five is None and weekly is None:
        return "unknown"
    return f"{pct(five)} / {pct(weekly)}"


def account_reset_note(label: str, resets: dict[str, float], now: float | None = None) -> str:
    """Documented tail for one stored account: 'resets in 4h 12m' or ''."""
    if not resets:
        return ""
    return reset_note(resets, now)


def last_reset_text(resets: dict[str, float], now: float | None = None) -> str:
    """'3h ago' style text for reset stamps already in the past (kept module-level for tests)."""
    now = time.time() if now is None else now
    stamps = [value for value in resets.values() if isinstance(value, (int, float))]
    if not stamps:
        return "-"
    delta = now - max(stamps)
    if delta <= 0:
        return next_reset(resets, now)
    hours = int(delta // 3600)
    minutes = int(delta // 60)
    if hours >= 1:
        return f"{hours}h ago"
    return f"{minutes}m ago"


def describe_rotation_mode(mode: str) -> str:
    """One-line effect summary for a rotation mode."""
    return {
        "off": "single account (no rotation)",
        "quota": "pick healthiest account by remaining quota",
        "round_robin": "cycle through accounts",
        "fixed": "always use the active account",
    }.get(mode, "")


def ignite_effect(active: bool) -> str:
    """One-line effect summary for quota ignition."""
    if active:
        return "5h windows can be kicked awake on idle accounts, trading a bit of quota"
    return "idle 5h windows are left untouched (no extra quota spent)"


def active_store_name(registry: dict[str, Any] | None = None) -> str:
    """Account label currently selected in the registry, or '' for the host default."""
    if registry is None:
        registry = load_registry()
    active = registry.get("active_account")
    if isinstance(active, dict):  # legacy shape written by early companions
        active = active.get("label")
    label = str(active or "").strip()
    if label == HOST_LABEL:
        return ""
    return label


def find_account_dir(label: str) -> Path | None:
    """Best-effort location of a stored account dir matching a label (registry first, then disk)."""
    wanted = (label or "").strip()
    if not wanted or wanted == HOST_LABEL:
        return None
    for entry in registry_accounts():
        current = (str(entry.get("label") or "").strip() or Path(str(entry.get("home_dir") or "")).name).strip()
        if current == wanted:
            home = Path(str(entry.get("home_dir") or "")).expanduser()
            return home if str(home) else None
    root = accounts_dir()
    if root.is_dir():
        child = root / wanted
        if child.is_dir():
            return child
    return None


def save_registry(mutator) -> dict[str, Any]:
    """Read-modify-write the registry through `mutator(data) -> data`, preserving unknown fields."""
    path = registry_file()
    data = load_registry()
    updated = mutator(data)
    if not isinstance(updated, dict):
        updated = data
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(updated, indent=2, sort_keys=True))
    except Exception as exc:
        raise RuntimeError(f"cannot write registry {path}: {exc}") from exc
    return updated


def set_active_label(label: str | None) -> dict[str, Any]:
    """Persist `active_account` as the account LABEL string.

    Contract with the provider plugin: it stores and reads `active_account` as a
    plain label string (see `set_active` / `pick_account` in
    antigravity-subscription-directsdk). None or 'host' clears it back to the
    host default.
    """
    chosen = (label or "").strip() or None
    if chosen == HOST_LABEL:
        chosen = None

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        if chosen is None:
            data.pop("active_account", None)
        else:
            data["active_account"] = chosen
        return data

    return save_registry(mutate)


def save_rotation_mode(mode: str) -> dict[str, Any]:
    """Persist `rotation_mode` to the registry, preserving unknown fields."""

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["rotation_mode"] = mode
        return data

    return save_registry(mutate)


def save_quota_ignition(enabled: bool) -> dict[str, Any]:
    """Persist `quota_ignition` to the registry, preserving unknown fields."""

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["quota_ignition"] = enabled
        return data

    return save_registry(mutate)


def parse_reset_time(value: Any) -> float | None:
    """Parse one agy `reset_time` bucket value into a unix timestamp (UTC seconds).

    Accepts a unix timestamp (number or numeric string, seconds or milliseconds)
    or an RFC 3339 "Z"/offset timestamp. Returns None when it cannot be parsed.
    Kept at module scope for unit tests.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds <= 0:
            return None
        return seconds / 1000.0 if seconds > 1e11 else seconds
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        seconds = float(text)
        if seconds <= 0:
            return None
        return seconds / 1000.0 if seconds > 1e11 else seconds
    cleansed = text.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        stamp = datetime.fromisoformat(cleansed)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def format_time_remaining(seconds: float) -> str:
    """Human short countdown: 'in 4h 12m', 'in 3d 2h', 'overdue', 'now'."""
    text = "in "
    try:
        seconds_int = int(seconds)
    except (TypeError, ValueError):
        return "?"
    if seconds_int <= 0:
        return "now"
    days, rem = divmod(seconds_int, 86400)
    if days >= 1:
        hours = rem // 3600
        return f"{text}{days}d {hours}h" if hours else f"{text}{days}d"
    if seconds_int < 60:
        return f"{text}{seconds_int}s"
    minutes, _ = divmod(rem, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f"{text}{hours}h" if minutes == 0 else f"{text}{hours}h {minutes}m"
    return f"{text}{minutes}m"


def reset_pair(resets: dict[str, float], key: str, now: float | None = None) -> str:
    """'in 4h 2m / in 3d' — nearest countdown for the group's 5h bucket and its weekly one."""
    now = time.time() if now is None else now
    five, weekly = resets.get(f"{key}_5h"), resets.get(f"{key}_weekly")
    if five is None and weekly is None:
        return "-"
    five_text = format_time_remaining(five - now) if five is not None else "?"
    weekly_text = format_time_remaining(weekly - now) if weekly is not None else "?"
    return f"{five_text} / {weekly_text}"


def next_reset(resets: dict[str, float], now: float | None = None) -> str:
    """'in 4h 12m' — countdown to the nearest reset bucket across all groups."""
    now = time.time() if now is None else now
    stamps = [value for value in resets.values() if isinstance(value, (int, float))]
    if not stamps:
        return "-"
    return format_time_remaining(min(stamps) - now)


def reset_note(resets: dict[str, float], now: float | None = None) -> str:
    """'resets in 4h 12m' or '' — short tail used after tables."""
    stamp = next_reset(resets, now)
    return "" if stamp == "-" else f"resets {stamp}"


def account_key(home: Path | str) -> str:
    """Stable key matching an account to its stored dir: 'host' for the home dir."""
    resolved = str(Path(home).expanduser().resolve())
    if resolved == str(Path.home().resolve()):
        return HOST_LABEL
    return resolved


def reset_pair_with_val(resets: dict[str, float], key: str, now: float | None = None) -> str:
    """Same as reset_pair, but only the group's nearest countdown ('12h'), used for narrow tables."""
    text = reset_pair(resets, key, now)
    if " / " in text:
        return text.split(" / ")[0]
    return text
