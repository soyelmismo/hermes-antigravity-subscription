"""The agent agy runs as when it serves Hermes.

Left alone, agy runs its default agent: a system prompt that names the model
"Antigravity" and hands it run_command, write_file, edit_file and a dozen more
tools. That prompt sits ahead of everything the plugin sends, so the plugin's
preamble can only argue with it, and which side wins is per-model (issue #16:
claude-sonnet-4-6 answers "you are currently talking to Antigravity" and goes
native with a large toolset; gemini-3.6-flash goes native with one tool).

agy also runs agents defined in Markdown. One declared with
``excludeDefaultComponents: true`` (agy 1.2.1) carries no default prompt
sections and no built-in tools, so there is no persona left to argue with and
no native tool to reach for except the one the definition lists. Measured
through this plugin on agy 1.5.16 with gemini-3.8-flash-low, a one-line
Hermes system prompt and the question "what is your name, which native tools
can you call":

    default agent   "I am Antigravity", 15 tools            10,749 input tokens
    this agent      "My name is Hermes", read_file,
                    manage_task                              1,218 input tokens

``read_file`` stays because agy attaches media pixels through it (see
stream._is_permitted_native_media_step). agy adds ``manage_task`` on its own;
stream.py neutralizes it like any other native step.

Everything here degrades to the default agent: an agy older than the version
this was verified on, an opt-out, an unwritable home, or a spawn agy rejects
all leave the spawn arguments exactly as they were before this module existed.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from pathlib import Path

try:
    from .process import _no_window
except ImportError:
    from process import _no_window

logger = logging.getLogger(__name__)

AGENT_NAME = "hermes-inference-backend"

# The agy this agent was run on (1.5.16, macOS); nothing older was. The changelog
# puts the pieces earlier -- excludeDefaultComponents in 1.2.1, --agent finding
# agents in headless runs in 1.2.11 -- but agy 1.2 called the read tool
# view_file (its 1.2.2 notes), and a tool name the registry lacks fails the
# spawn rather than falling back. Lower the floor only together with a run on
# that version.
MIN_AGY_VERSION = (1, 5, 16)

_DISABLE_ENV = "ANTIGRAVITY_DISABLE_CUSTOM_AGENT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# agy reads the definition once, at spawn. ``name`` and ``description`` are
# required; a YAML type error skips the whole file.
_AGENT_MARKDOWN = f"""---
name: {AGENT_NAME}
description: "Inference backend for Hermes Agent: no agy persona prompt and no agy tools beyond the media read."
excludeDefaultComponents: true
inheritMcp: false
subagent: false
mainAgent: true
tools:
  - read_file
---
# Role
You are the language model behind Hermes Agent, used as an inference backend. Each user message carries Hermes' system instructions, its tool schemas and the conversation. Follow them.

# Tools
Hermes executes every tool on the host. Ask for a tool only in the form the user message specifies. Your one native tool, read_file, exists to load a media file that a message references as [Image](path), [Video](path) or [Audio](path), and for nothing else.
"""

# What agy answers when it cannot run the agent it was asked for (verbatim
# from 1.5.16: a `result` event with status ERROR, the same text on stderr).
# The second is a tool name agy's registry no longer has.
_REJECTION_SIGNATURES = (
    "unknown agent",
    "failed to construct executor",
)

_lock = threading.Lock()
_versions: dict[str, tuple[int, ...] | None] = {}
_rejected = False


def agy_version(command: str) -> tuple[int, ...] | None:
    """The version ``command --version`` prints, or None when it cannot be read."""
    with _lock:
        if command in _versions:
            return _versions[command]
    version: tuple[int, ...] | None = None
    try:
        res = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            **_no_window(),
        )
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(res.stdout))
        if match:
            version = tuple(int(part) for part in match.groups())
    except Exception as exc:
        logger.debug("Antigravity version unavailable: %s", exc)
    with _lock:
        _versions[command] = version
    return version


def _opted_out() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY


def agent_path(isolated_home: Path | str) -> Path:
    """Where agy looks for a user-level agent under the plugin's isolated HOME."""
    return Path(isolated_home) / ".gemini" / "config" / "agents" / AGENT_NAME / "agent.md"


def custom_agent_args(command: str, isolated_home: Path | str) -> list[str]:
    """``--agent`` arguments for an agy spawn, or ``[]`` to run agy's default agent.

    Writes the definition under the isolated HOME on the way, so a spawn never
    names an agent agy cannot find. The file lives beside agy's own state, not
    in the working directory, and goes away with the session's temp dir.
    """
    if _rejected or _opted_out():
        return []
    version = agy_version(command)
    if version is None or version < MIN_AGY_VERSION:
        return []
    path = agent_path(isolated_home)
    try:
        if not path.is_file() or path.read_text(encoding="utf-8") != _AGENT_MARKDOWN:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_AGENT_MARKDOWN, encoding="utf-8")
    except OSError as exc:
        logger.debug("Antigravity custom agent not written: %s", exc)
        return []
    return ["--agent", AGENT_NAME]


def note_custom_agent_rejection(detail: object) -> bool:
    """Stop naming the agent for the rest of the process if agy refused to run it.

    A later agy may rename the tool the definition lists or change how agents
    load. The turn that hit this has already failed; Hermes retries it, and the
    retry spawns agy's default agent, which is the behavior before this module.
    """
    global _rejected
    lowered = str(detail or "").lower()
    if not any(signature in lowered for signature in _REJECTION_SIGNATURES):
        return False
    with _lock:
        if _rejected:
            return True
        _rejected = True
    logger.warning(
        "agy refused the %s agent (%s); using agy's default agent for the rest of this process.",
        AGENT_NAME,
        str(detail).strip()[:200],
    )
    return True
