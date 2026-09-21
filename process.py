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
            if p.is_file() and (os.name == "nt" or os.access(val, os.X_OK)):
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
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)

    return "agy"


def resolve_real_token_path() -> Path | None:
    """Locate the authentic Antigravity OAuth token on the host."""
    token_dir = os.getenv("ANTIGRAVITY_CONFIG_DIR", "").strip()
    if token_dir:
        p = Path(token_dir) / "antigravity-oauth-token"
        if p.is_file():
            return p
    p = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    if p.is_file():
        return p
    fallback = Path("/root/.gemini/antigravity-cli/antigravity-oauth-token")
    if fallback.is_file():
        return fallback
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
    if real_token and real_token.is_file():
        isolated_token = isolated_gemini_dir / "antigravity-oauth-token"
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
