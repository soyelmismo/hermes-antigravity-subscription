"""Cross-platform process, OS support, and isolated workspace management for Antigravity."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

# Marker scheme for Antigravity local provider
AGY_MARKER_BASE_URL = "agy://local"

# Known OAuth token basenames. agy1.2 renamed the fallback file from
# antigravity-oauth-token to jetski-standalone-oauth-token (issue #1),
# so both must resolve. Order is new-first: an upgraded user (1.1 -> 1.2)
# has BOTH files in the same directory, because 1.2 does not remove the
# old one, and 1.2 reads only the new name — so when both coexist the new
# name is authoritative. Resolving the stale legacy file there would
# authenticate (size > 10) a token that agy 1.2 ignores. Legacy-only 1.1
# users are unaffected: the new file simply does not exist and the scan
# falls through to the legacy name.
_TOKEN_FILENAMES = ("jetski-standalone-oauth-token", "antigravity-oauth-token")


def _is_existing_file(path: str | Path) -> bool:
    """True if path is a regular file, treating any OS error as absent.

    Candidate discovery walks directories that may be unreadable to the
    current user (Path.home() of another account, /root under a
    non-root runner). pathlib's is_file() only swallows part of the
    OSError family, so a PermissionError on stat() escaped and aborted
    the whole scan instead of just skipping that candidate.
    """
    try:
        return Path(path).is_file()
    except OSError:
        return False


def _own_process_group() -> dict[str, Any]:
    """Popen kwargs that put native (and any child processes it spawns) in a group we can kill cleanly."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill process and every descendant: taskkill /F /T on Windows, killpg on POSIX."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — the nt branch above never reaches this line
    except (ProcessLookupError, PermissionError, AttributeError):
        with contextlib.suppress(Exception):
            proc.kill()


def terminate_process(proc: subprocess.Popen) -> None:
    """Attempt graceful termination, falling back to process-tree kill."""
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        _kill_process_tree(proc)


def resolve_agy_command() -> str:
    """Find the path to the official `agy` binary across platforms."""
    for var in ("ANTIGRAVITY_COMMAND", "AGY_CLI_PATH", "ANTIGRAVITY_CLI_PATH"):
        if val := os.getenv(var, "").strip():
            p = Path(val)
            if _is_existing_file(p) and (os.name == "nt" or os.access(val, os.X_OK)):
                return val

    # Check PATH (shutil.which checks PATHEXT on Windows, e.g. agy.exe)
    if path := shutil.which("agy"):
        return path

    binary_name = "agy.exe" if os.name == "nt" else "agy"
    candidates = [
        Path.home() / ".gemini" / "antigravity-cli" / "bin" / binary_name,
        Path.home() / ".local" / "bin" / binary_name,
        Path("/root/.local/bin") / binary_name,
        Path("/usr/local/bin") / binary_name,
        Path("/usr/bin") / binary_name,
    ]
    if os.name == "nt":
        if localappdata := os.getenv("LOCALAPPDATA"):
            candidates.append(Path(localappdata) / "Programs" / "agy" / binary_name)
            candidates.append(Path(localappdata) / "Microsoft" / "WinGet" / "Links" / binary_name)

    for candidate in candidates:
        if _is_existing_file(candidate) and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)

    return "agy"


def resolve_real_token_path() -> Path | None:
    """Locate the authentic Antigravity OAuth token on the host.

    ANTIGRAVITY_CONFIG_DIR is an override, not a hint: when it is set to a
    directory without a token, the explicit configuration wins and the
    implicit locations are not consulted. Falling through anyway would let a
    stray /root token satisfy auth for a user who deliberately pointed
    ANTIGRAVITY_CONFIG_DIR somewhere else.
    """
    token_dir = os.getenv("ANTIGRAVITY_CONFIG_DIR", "").strip()
    if token_dir:
        config_path = Path(token_dir)
        for filename in _TOKEN_FILENAMES:
            explicit = config_path / filename
            if _is_existing_file(explicit):
                return explicit
        return None

    home_base = Path.home() / ".gemini" / "antigravity-cli"
    root_base = Path("/root/.gemini/antigravity-cli")
    candidates = [home_base / name for name in _TOKEN_FILENAMES]
    # Last resort for containers/sudo contexts where HOME does not point
    # at the account that ran `agy`. All home candidates win over any
    # /root fallback so a stale /root legacy token never beats the user's
    # current token. Skip the extra stat when HOME already is /root.
    if root_base != home_base:
        candidates += [root_base / name for name in _TOKEN_FILENAMES]
    for candidate in candidates:
        if _is_existing_file(candidate):
            return candidate
    return None


def is_authenticated() -> bool:
    """Verify that the user has an active Antigravity OAuth session.
    
    Zero-Exfiltration compliance: We only verify file presence and non-zero
    size. We NEVER read, parse, or transmit the token contents.
    """
    token_path = resolve_real_token_path()
    if not token_path:
        return False
    try:
        return token_path.is_file() and token_path.stat().st_size > 10
    except OSError:
        return False


def setup_isolated_home(cwd: Path | str) -> tuple[Path, Path]:
    """Create isolated HOME inside cwd and link authentic oauth token."""
    isolated_home = Path(cwd) / "home"
    isolated_gemini_dir = isolated_home / ".gemini" / "antigravity-cli"
    isolated_gemini_dir.mkdir(parents=True, exist_ok=True)

    real_token = resolve_real_token_path()
    if real_token and _is_existing_file(real_token):
        # Reused cwd: a previous run may have selected the OTHER known
        # basename (e.g. a pre-upgrade agy1.1 run linked
        # antigravity-oauth-token). Such a leftover link would linger next
        # to the fresh one, so drop it — but only if it is a symlink we
        # created; never unlink a real file that could be user data.
        for stale_name in _TOKEN_FILENAMES:
            if stale_name == real_token.name:
                continue
            stale_link = isolated_gemini_dir / stale_name
            if os.path.islink(stale_link):
                with contextlib.suppress(OSError):
                    stale_link.unlink()
        # Preserve the selected basename so the agy1.2 filename keeps
        # working inside the isolated home (zero secret parsing).
        isolated_token = isolated_gemini_dir / real_token.name
        # Reused cwd: the link under the selected basename may itself be a
        # leftover pointing at a previous run's source. If that source is
        # stale but still present, the exists() guard below would skip
        # recreation and the child would use the old token; if it is
        # dangling (previous run's temp source deleted), symlink raises
        # FileExistsError and the copy2 fallback opens through the dangling
        # link and fails, leaving no usable token. Always relink to the
        # CURRENT real token. A real regular file is left untouched: it can
        # be a valid copy2 hardlink-failure artifact and could be user data.
        if os.path.islink(isolated_token):
            with contextlib.suppress(OSError):
                isolated_token.unlink()
        if not isolated_token.exists():
            try:
                os.symlink(real_token, isolated_token)
            except OSError:
                try:
                    os.link(real_token, isolated_token)
                except OSError:
                    with contextlib.suppress(OSError):
                        shutil.copy2(real_token, isolated_token)

    return isolated_home, isolated_gemini_dir


def build_child_env(isolated_home: Path | str) -> dict[str, str]:
    """Construct child environment isolating home and session storage on POSIX and Windows."""
    env = dict(os.environ)
    home_str = str(isolated_home)
    env["HOME"] = home_str
    # Windows: Go's os.UserHomeDir() reads USERPROFILE then HOMEDRIVE+HOMEPATH
    env["USERPROFILE"] = home_str
    if "HOMEPATH" in env:
        env["HOMEPATH"] = home_str
    return env


def _check_early_quota_error(
    gemini_dir: Path | str | None,
    min_mtime: float | None = None,
) -> str | None:
    """Check isolated agy logs for early RESOURCE_EXHAUSTED / 429 quota exhaustion.

    agy internally retries 429s up to 8 times with exponential backoff (~140s)
    without emitting anything to stdout. Detecting this early allows fast failover.
    """
    if not gemini_dir:
        return None
    log_dir = Path(gemini_dir) / "log"
    if not log_dir.is_dir():
        return None

    try:
        log_files = sorted(
            log_dir.glob("cli-*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not log_files:
            return None

        latest_log = log_files[0]
        if min_mtime is not None:
            # Allow 2.0s margin for filesystem timestamp resolution
            if latest_log.stat().st_mtime < (min_mtime - 2.0):
                return None

        with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            content = f.read()

        for line in reversed(content.splitlines()):
            if "RESOURCE_EXHAUSTED" in line and (
                "Individual quota reached" in line
                or "quota exceeded" in line.lower()
                or "code 429" in line
            ):
                start = line.find("RESOURCE_EXHAUSTED")
                if start != -1:
                    detail = line[start:]
                    retrying_idx = detail.rfind("), retrying in")
                    if retrying_idx != -1:
                        detail = detail[:retrying_idx]
                    if detail.count("(") < detail.count(")"):
                        detail = detail.rstrip(")")
                    return detail.strip()
                return line.strip()
    except Exception:
        pass
    return None

