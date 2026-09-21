"""Antigravity Subscription DirectSDK client for Hermes Agent.

Drives the official Antigravity CLI (`agy`) as an external process in
stream-json mode to provide request-scoped completions using the user's
existing Antigravity / Gemini subscription quota.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

# Base scheme marker for Antigravity
AGY_MARKER_BASE_URL = "agy://local"
_DEFAULT_TIMEOUT_SECONDS = 300.0

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}

_PROMPT_PREAMBLE = (
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "You do not possess any local execution tools in this mode. You MUST NOT attempt to invoke native agent tools (such as run_command, write_to_file, view_file, etc.).",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output.",
    "- Each tool call must be a JSON object containing 'id', 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}.",
    "- 'arguments' must be a JSON-encoded string containing the function arguments.",
    "- Do NOT execute local shell commands or file operations directly; only output <tool_call> tags so Hermes can execute them safely.",
    "- If no tool is needed, respond naturally with standard text.",
)

_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
]

_MODEL_ALIASES = {
    "default": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "gemini-flash": "gemini-3.8-flash",
    "gemini-3.8": "gemini-3.8-flash",
    "pro": "gemini-3.1-pro",
    "gemini-pro": "gemini-3.1-pro",
    "gemini-3.1": "gemini-3.1-pro",
    "sonnet": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6-thinking",
    "claude-opus": "claude-opus-4-6-thinking",
}


def _normalize_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    e = str(effort).strip().lower()
    if e in ("none", "off", "minimal"):
        return "low"
    if e in ("xhigh", "max", "ultra"):
        return "high"
    if e in ("low", "medium", "high"):
        return e
    return "medium"



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


def resolve_agy_command() -> str:
    """Find the path to the official `agy` binary."""
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


def is_authenticated() -> bool:
    """Verify that the user has an active Antigravity OAuth session.
    
    Zero-Exfiltration compliance: We only verify file presence and non-zero
    size. We NEVER read, parse, or transmit the token contents.
    """
    token_dir = os.getenv("ANTIGRAVITY_CONFIG_DIR", "").strip()
    if token_dir:
        token_path = Path(token_dir) / "antigravity-oauth-token"
    else:
        token_path = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        if not token_path.exists():
            token_path = Path("/root/.gemini/antigravity-cli/antigravity-oauth-token")

    try:
        return token_path.is_file() and token_path.stat().st_size > 10
    except OSError:
        return False


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        return str(content.get("content") or "").strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or "").strip())
                elif "content" in item:
                    parts.append(str(item.get("content") or "").strip())
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Assemble Hermes message history, system prompt, and tool schemas into a canonical prompt."""
    try:
        from agent.acp_openai_bridge import render_tool_bridge_sections
        tool_sections = render_tool_bridge_sections(tools, tool_choice)
    except Exception:
        tool_sections = []
        if tools:
            tool_sections.append(
                "Available tools (OpenAI function schema):\n"
                + json.dumps(tools, ensure_ascii=False)
            )

    sections: list[str] = [*_PROMPT_PREAMBLE, *tool_sections]
    transcript: list[str] = []

    for message in (m for m in messages if isinstance(m, dict)):
        role = str(message.get("role") or "unknown").strip().lower()
        rendered_content = _render_message_content(message.get("content"))

        if role == "tool":
            tool_id = str(message.get("tool_call_id") or message.get("name") or "tool").strip()
            transcript.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue

        if role == "assistant":
            parts = []
            if rendered_content:
                parts.append(rendered_content)
            if tool_calls := message.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        parts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if parts:
                transcript.append(f"Assistant:\n" + "\n".join(parts))
            continue

        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            transcript.append(f"{label}:\n{rendered_content}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _messages_match_prefix(history: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]]) -> bool:
    """Return True if incoming messages strictly extend history as a continuation."""
    if not history or len(incoming) <= len(history):
        return False
    for i, h_msg in enumerate(history):
        inc_msg = incoming[i]
        if not isinstance(inc_msg, dict) or not isinstance(h_msg, dict):
            return False
        if inc_msg.get("role") != h_msg.get("role"):
            return False
        if inc_msg.get("content") != h_msg.get("content"):
            return False
    return True


def _format_delta_prompt(new_messages: Sequence[dict[str, Any]]) -> str:
    """Format only the incremental messages in an ongoing multi-turn interaction."""
    parts: list[str] = []
    for msg in new_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        rendered_content = _render_message_content(msg.get("content"))
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("name") or "tool").strip()
            parts.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue
        if role == "assistant":
            subparts = []
            if rendered_content:
                subparts.append(rendered_content)
            if tool_calls := msg.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        subparts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if subparts:
                parts.append(f"Assistant:\n" + "\n".join(subparts))
            continue
        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            parts.append(f"{label}:\n{rendered_content}")

    parts.append("Continue the conversation from the latest tool result.")
    return "\n\n".join(s.strip() for s in parts if s and s.strip())


_TOOL_CALL_PREFIXES = tuple(
    "<tool_call>"[:i] for i in range(len("<tool_call>"), 0, -1)
)


def _longest_tool_call_prefix_match(text: str) -> int:
    """Return the length of the longest suffix of `text` that matches a prefix of `<tool_call>`."""
    for prefix in _TOOL_CALL_PREFIXES:
        if text.endswith(prefix):
            return len(prefix)
    return 0


def _parse_tool_block(block: str) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text)."""
    calls = []
    cleaned_text = ""
    try:
        from agent.acp_openai_bridge import extract_tool_calls_from_text
        calls, cleaned_text = extract_tool_calls_from_text(block)
    except Exception:
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", block, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                fn = obj.get("function", {})
                call_id = obj.get("id") or "call_1"
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", "{}")
                if not isinstance(fn_args, str):
                    fn_args = json.dumps(fn_args, ensure_ascii=False)
                calls = [
                    SimpleNamespace(
                        id=call_id,
                        type="function",
                        function=SimpleNamespace(name=fn_name, arguments=fn_args),
                    )
                ]
                cleaned_text = block[:m.start()] + block[m.end():]
            except Exception:
                cleaned_text = block
        else:
            cleaned_text = block

    deltas = []
    for i, call in enumerate(calls):
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args = getattr(fn, "arguments", "{}") if fn else "{}"
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        deltas.append(
            SimpleNamespace(
                index=i,
                id=getattr(call, "id", f"call_{i+1}"),
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )
        )
    return deltas, cleaned_text.strip()


class AntigravityStream(Iterator[Any]):
    """Streaming iterator that yields OpenAI-compatible ChatCompletionChunk objects
    progressively as tokens arrive from Antigravity CLI's stream-json output.
    """

    response: Any = None  # Mock response attribute for Hermes Relay compatibility

    def __init__(
        self,
        *,
        proc: subprocess.Popen,
        client: AntigravityClient,
        model: str,
        timeout: float,
        tools: list[dict[str, Any]] | None = None,
        is_worker: bool = False,
        worker_lock: threading.Lock | None = None,
        messages: list[dict[str, Any]] | None = None,
    ):
        self.proc = proc
        self.client = client
        self.model = model
        self.timeout = timeout
        self.has_tools = bool(tools)
        self.is_worker = is_worker
        self.worker_lock = worker_lock
        self.messages = messages
        self.conversation_id = ""
        self._closed = False
        self._interrupted = False
        self._finished = False
        self._generator = self._stream_generator()

    def __iter__(self) -> "AntigravityStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration:
            self._closed = True
            if not self.is_worker:
                self.close()
            raise
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.is_worker:
            if not self._finished:
                self._interrupted = True
                self.client._terminate_worker()
            if self.worker_lock and self.worker_lock.locked():
                with contextlib.suppress(Exception):
                    self.worker_lock.release()
        else:
            with self.client._lock:
                self.client._active_processes.discard(self.proc)
            self.client._terminate_process(self.proc)

    def _make_chunk(
        self,
        *,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
            choices=[choice],
            model=self.model,
            usage=None,
        )

    def _stream_generator(self) -> Iterator[Any]:
        deadline = time.monotonic() + self.timeout
        text_buffer = ""
        usage_data: dict[str, Any] = {}
        has_tool_calls = False
        has_content = False
        in_tool_call = False
        error_msg = ""
        status = ""
        success = False

        try:
            while time.monotonic() < deadline:
                line = self.proc.stdout.readline() if self.proc.stdout else ""
                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue

                line = line.strip()
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                event_type = event.get("event")
                if not self.conversation_id:
                    self.conversation_id = event.get("conversation_id", "")

                if event_type == "init":
                    init_data = event.get("init", {})
                    if not self.conversation_id:
                        self.conversation_id = init_data.get("conversation_id", "")

                elif event_type == "step_update":
                    step = event.get("step_update", {})
                    if not self.conversation_id:
                        self.conversation_id = step.get("conversation_id", "")
                    if step.get("step_type") == "tool":
                        logger.warning(
                            "Antigravity attempted native tool invocation '%s'; neutralizing to prevent host execution.",
                            step.get("tool_name"),
                        )
                        self.close()
                        break
                    if "usage" in step:
                        usage_data = step["usage"]

                    text_delta = step.get("text_delta")
                    if text_delta:
                        if not self.has_tools:
                            has_content = True
                            yield self._make_chunk(content=text_delta)
                        else:
                            text_buffer += text_delta
                            while text_buffer:
                                if in_tool_call:
                                    end_idx = text_buffer.find("</tool_call>")
                                    if end_idx != -1:
                                        full_end = end_idx + len("</tool_call>")
                                        block = text_buffer[:full_end]
                                        text_buffer = text_buffer[full_end:]
                                        in_tool_call = False
                                        parsed_calls, extra_text = _parse_tool_block(block)
                                        if parsed_calls:
                                            has_tool_calls = True
                                            for call_delta in parsed_calls:
                                                yield self._make_chunk(tool_calls=[call_delta])
                                        if extra_text:
                                            has_content = True
                                            yield self._make_chunk(content=extra_text)
                                    else:
                                        # Inside <tool_call>, continue accumulating
                                        break
                                else:
                                    idx = text_buffer.find("<tool_call")
                                    if idx != -1:
                                        if idx > 0:
                                            safe_text = text_buffer[:idx]
                                            has_content = True
                                            yield self._make_chunk(content=safe_text)
                                        text_buffer = text_buffer[idx:]
                                        in_tool_call = True
                                    else:
                                        k = _longest_tool_call_prefix_match(text_buffer)
                                        if k > 0:
                                            safe_text = text_buffer[:-k]
                                            if safe_text:
                                                has_content = True
                                                yield self._make_chunk(content=safe_text)
                                            text_buffer = text_buffer[-k:]
                                            break
                                        else:
                                            has_content = True
                                            yield self._make_chunk(content=text_buffer)
                                            text_buffer = ""

                elif event_type == "result":
                    res = event.get("result", {})
                    if not self.conversation_id:
                        self.conversation_id = res.get("conversation_id", "")
                    status = res.get("status", "")
                    if "usage" in res:
                        usage_data = res["usage"]
                    if "error" in res:
                        error_msg = res["error"]
                    success = (status != "ERROR")
                    break

            # Handle remaining buffer at stream end
            if text_buffer:
                if self.has_tools:
                    parsed_calls, extra_text = _parse_tool_block(text_buffer)
                    if parsed_calls:
                        has_tool_calls = True
                        for call_delta in parsed_calls:
                            yield self._make_chunk(tool_calls=[call_delta])
                    if extra_text:
                        has_content = True
                        yield self._make_chunk(content=extra_text)
                    elif not parsed_calls:
                        has_content = True
                        yield self._make_chunk(content=text_buffer)
                else:
                    has_content = True
                    yield self._make_chunk(content=text_buffer)

            if not self.is_worker:
                # Wait for process exit cleanly
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.client._terminate_process(self.proc)

                stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
                returncode = self.proc.poll() or 0

                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")

                if not has_tool_calls and not has_content and returncode != 0:
                    err_detail = error_msg or stderr_out.strip() or f"Process exited with return code {returncode}"
                    raise RuntimeError(f"Antigravity execution failed: {err_detail}")
            else:
                if status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {error_msg}")

            # Yield finish reason chunk
            finish_reason = "tool_calls" if has_tool_calls else "stop"
            yield self._make_chunk(finish_reason=finish_reason)

            # Yield usage chunk
            input_tokens = int(usage_data.get("input_tokens", 0) or 0)
            output_tokens = int(usage_data.get("output_tokens", 0) or 0)
            total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens) or 0)
            cached_tokens = int(usage_data.get("cache_read_tokens", 0) or 0)

            usage = SimpleNamespace(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            )
            yield SimpleNamespace(
                id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
                choices=[],
                model=self.model,
                usage=usage,
            )
            self._finished = True
        finally:
            if self.is_worker:
                if success and not self._interrupted:
                    self.client._update_worker_history(self.messages)
                    if self.worker_lock and self.worker_lock.locked():
                        with contextlib.suppress(Exception):
                            self.worker_lock.release()
                else:
                    self.close()
            else:
                self.close()


class AntigravityClient:
    """OpenAI-compatible client facade that drives Antigravity CLI."""

    # Instruct Hermes not to wrap this client in wire transports or async adapters
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "antigravity-directsdk"
        self.base_url = base_url or AGY_MARKER_BASE_URL
        self._command = command or resolve_agy_command()
        self._args = list(args or ["--output-format", "stream-json", "--disable-slash-commands"])
        self._temp_dir = None
        if cwd:
            self._cwd = cwd
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_")
            self._cwd = self._temp_dir.name

        # Isolate agy state and session index from the user's real ~/.gemini/antigravity-cli
        self._isolated_home = Path(self._cwd) / "home"
        self._isolated_gemini_dir = self._isolated_home / ".gemini" / "antigravity-cli"
        self._isolated_gemini_dir.mkdir(parents=True, exist_ok=True)

        real_token = self._resolve_real_token_path()
        if real_token and real_token.is_file():
            isolated_token = self._isolated_gemini_dir / "antigravity-oauth-token"
            if not isolated_token.exists():
                try:
                    os.symlink(real_token, isolated_token)
                except OSError:
                    try:
                        os.link(real_token, isolated_token)
                    except OSError:
                        with contextlib.suppress(OSError):
                            shutil.copy2(real_token, isolated_token)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        self._active_processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._worker_proc: subprocess.Popen | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_history: list[dict[str, Any]] = []
        self._worker_lock = threading.Lock()

    @staticmethod
    def _resolve_real_token_path() -> Path | None:
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

    def _resolve_model_and_effort(
        self,
        model: str | None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        m = str(model or "gemini-3.8-flash").strip()
        m = _MODEL_ALIASES.get(m.lower(), m)

        # Check if the model already contains an explicit effort suffix
        base_model = m
        suffix_effort = None
        for suffix, eff in (("-high", "high"), ("-medium", "medium"), ("-low", "low")):
            if m.endswith(suffix):
                suffix_effort = eff
                base_model = m[:-len(suffix)]
                break

        # Resolve effort: explicit argument > model suffix > config setting > default
        effort = _normalize_effort(reasoning_effort)
        if not effort and suffix_effort:
            effort = suffix_effort
        if not effort:
            try:
                from hermes_cli.config import load_config_readonly
                cfg_effort = load_config_readonly().get("agent", {}).get("reasoning_effort")
                effort = _normalize_effort(cfg_effort)
            except Exception:
                pass
        if not effort:
            effort = "medium"

        # Map base_model + effort to concrete agy model ID
        if base_model == "gemini-3.1-pro":
            # gemini-3.1-pro only has -low and -high in agy
            concrete_effort = "low" if effort == "low" else "high"
            target_model = f"{base_model}-{concrete_effort}"
            return target_model, concrete_effort
        elif base_model.startswith("gemini-") and "flash" in base_model:
            concrete_effort = effort if effort in ("low", "medium", "high") else "medium"
            target_model = f"{base_model}-{concrete_effort}"
            return target_model, concrete_effort
        elif suffix_effort and effort != suffix_effort:
            # Suffix was in model ID but user requested different reasoning effort
            concrete_effort = effort if effort in ("low", "medium", "high") else "medium"
            target_model = f"{base_model}-{concrete_effort}"
            return target_model, concrete_effort
        else:
            return m, effort

    def _child_env(self) -> dict[str, str]:
        """Construct child environment isolating home and session storage on POSIX and Windows."""
        env = dict(os.environ)
        home_str = str(self._isolated_home)
        env["HOME"] = home_str
        # Windows: Go's os.UserHomeDir() reads USERPROFILE then HOMEDRIVE+HOMEPATH
        env["USERPROFILE"] = home_str
        if "HOMEPATH" in env:
            env["HOMEPATH"] = home_str
        return env

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            _kill_process_tree(proc)

    def __enter__(self) -> "AntigravityClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.is_closed = True
            self._terminate_worker_locked()
            procs = tuple(self._active_processes)
            self._active_processes.clear()
        for proc in procs:
            self._terminate_process(proc)
        if self._temp_dir is not None:
            with contextlib.suppress(Exception):
                self._temp_dir.cleanup()
            self._temp_dir = None

    def _terminate_worker_locked(self) -> None:
        if self._worker_proc is not None:
            proc = self._worker_proc
            self._worker_proc = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_history = []
            self._active_processes.discard(proc)
            self._terminate_process(proc)

    def _terminate_worker(self) -> None:
        with self._lock:
            self._terminate_worker_locked()

    def _update_worker_history(self, messages: list[dict[str, Any]] | None) -> None:
        with self._lock:
            self._worker_history = list(messages or [])

    def _get_or_spawn_worker(self, model: str, effort: str | None) -> subprocess.Popen:
        with self._lock:
            if (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
                and self._worker_model == model
                and self._worker_effort == effort
            ):
                return self._worker_proc

            self._terminate_worker_locked()

            cmd_args = [self._command, "--input-format", "stream-json", *self._args]
            if model:
                cmd_args.extend(["--model", model])
            if effort:
                cmd_args.extend(["--effort", effort])

            proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd,
                env=self._child_env(),
                **_own_process_group(),
            )
            self._worker_proc = proc
            self._worker_model = model
            self._worker_effort = effort
            self._worker_history = []
            self._active_processes.add(proc)
            return proc

    def _collect_completion(
        self,
        *,
        proc: subprocess.Popen,
        model: str,
        timeout: float,
        tools: list[dict[str, Any]] | None = None,
        is_worker: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        text_deltas: list[str] = []
        final_response: str = ""
        conversation_id: str = ""
        usage_data: dict[str, Any] = {}
        error_msg: str = ""
        status: str = ""
        success = False

        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue

                line = line.strip()
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                event_type = event.get("event")
                if not conversation_id:
                    conversation_id = event.get("conversation_id", "")

                if event_type == "init":
                    init_data = event.get("init", {})
                    if not conversation_id:
                        conversation_id = init_data.get("conversation_id", "")
                elif event_type == "step_update":
                    step = event.get("step_update", {})
                    if not conversation_id:
                        conversation_id = step.get("conversation_id", "")
                    if step.get("step_type") == "tool":
                        logger.warning(
                            "Antigravity attempted native tool invocation '%s'; neutralizing to prevent host execution.",
                            step.get("tool_name"),
                        )
                        if is_worker:
                            self._terminate_worker()
                        else:
                            self._terminate_process(proc)
                        break
                    if "text_delta" in step:
                        text_deltas.append(step["text_delta"])
                    if "usage" in step:
                        usage_data = step["usage"]
                elif event_type == "result":
                    res = event.get("result", {})
                    if not conversation_id:
                        conversation_id = res.get("conversation_id", "")
                    status = res.get("status", "")
                    final_response = res.get("response", "")
                    if "usage" in res:
                        usage_data = res["usage"]
                    if "error" in res:
                        error_msg = res["error"]
                    success = (status != "ERROR")
                    break

            if not final_response and time.monotonic() >= deadline:
                if is_worker:
                    self._terminate_worker()
                else:
                    self._terminate_process(proc)
                raise TimeoutError(f"Antigravity CLI timed out after {timeout}s.")

            if not is_worker:
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._terminate_process(proc)

                stderr_out = proc.stderr.read() if proc.stderr else ""
                returncode = proc.poll() or 0

                if returncode != 0 and not final_response:
                    err_detail = error_msg or stderr_out.strip() or f"Process exited with return code {returncode}"
                    raise RuntimeError(f"Antigravity execution failed: {err_detail}")

            if status == "ERROR":
                raise RuntimeError(f"Antigravity model error: {error_msg}")

            if is_worker and success:
                self._update_worker_history(messages)

        finally:
            if not is_worker:
                with self._lock:
                    self._active_processes.discard(proc)
                self._terminate_process(proc)
            elif not success:
                self._terminate_worker()

        response_text = final_response if final_response else "".join(text_deltas)

        try:
            from agent.acp_openai_bridge import extract_tool_calls_from_text
            tool_calls, cleaned_text = extract_tool_calls_from_text(response_text)
        except Exception:
            tool_calls = []
            cleaned_text = response_text
            m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response_text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    fn = obj.get("function", {})
                    tool_calls = [
                        SimpleNamespace(
                            id=obj.get("id", "call_1"),
                            type="function",
                            function=SimpleNamespace(
                                name=fn.get("name", ""),
                                arguments=fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {})),
                            )
                        )
                    ]
                    cleaned_text = response_text[:m.start()] + response_text[m.end():]
                except Exception:
                    pass

        input_tokens = int(usage_data.get("input_tokens", 0) or 0)
        output_tokens = int(usage_data.get("output_tokens", 0) or 0)
        total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens) or 0)
        cached_tokens = int(usage_data.get("cache_read_tokens", 0) or 0)

        message = SimpleNamespace(
            role="assistant",
            content=cleaned_text.strip() if cleaned_text.strip() else None,
            tool_calls=tool_calls if tool_calls else None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )

        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

        usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        )

        return SimpleNamespace(
            id=conversation_id or f"agy-{int(time.time()*1000)}",
            choices=[choice],
            usage=usage,
            model=model,
        )

    def _run_oneshot_completion(
        self,
        *,
        model: str,
        effort: str | None,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        stream: bool,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages, model=model, tools=tools, tool_choice=tool_choice
        )
        cmd_args = [self._command, *self._args]
        if model:
            cmd_args.extend(["--model", model])
        if effort:
            cmd_args.extend(["--effort", effort])

        proc = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self._cwd,
            env=self._child_env(),
            **_own_process_group(),
        )

        try:
            if proc.stdin:
                proc.stdin.write(prompt_text + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except OSError:
            pass

        with self._lock:
            self._active_processes.add(proc)

        if stream:
            return AntigravityStream(
                proc=proc,
                client=self,
                model=model,
                timeout=timeout,
                tools=tools,
                is_worker=False,
            )
        else:
            return self._collect_completion(
                proc=proc,
                model=model,
                timeout=timeout,
                tools=tools,
                is_worker=False,
                messages=messages,
            )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        **extra_kwargs: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("AntigravityClient is closed.")

        if not is_authenticated():
            raise RuntimeError(
                "Antigravity CLI is not authenticated. Please run 'agy' in your terminal "
                "to log in with your Google account."
            )

        effort_param = reasoning_effort or extra_kwargs.get("reasoning_effort")
        resolved_model, effort = self._resolve_model_and_effort(model, effort_param)
        messages_list = list(messages or [])
        effective_timeout = float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else _DEFAULT_TIMEOUT_SECONDS

        worker_acquired = self._worker_lock.acquire(blocking=False)
        if worker_acquired:
            try:
                proc = self._get_or_spawn_worker(resolved_model, effort)
                with self._lock:
                    is_continuation = _messages_match_prefix(self._worker_history, messages_list)

                if is_continuation:
                    delta_msgs = messages_list[len(self._worker_history):]
                    prompt_payload = _format_delta_prompt(delta_msgs)
                else:
                    if self._worker_history:
                        self._terminate_worker()
                        proc = self._get_or_spawn_worker(resolved_model, effort)
                    prompt_payload = _format_messages_as_prompt(
                        messages_list, model=resolved_model, tools=tools, tool_choice=tool_choice
                    )

                event_msg = {"event": "user", "message": {"content": prompt_payload}}
                try:
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._terminate_worker()
                    proc = self._get_or_spawn_worker(resolved_model, effort)
                    prompt_payload = _format_messages_as_prompt(
                        messages_list, model=resolved_model, tools=tools, tool_choice=tool_choice
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()

                if stream:
                    return AntigravityStream(
                        proc=proc,
                        client=self,
                        model=resolved_model,
                        timeout=effective_timeout,
                        tools=tools,
                        is_worker=True,
                        worker_lock=self._worker_lock,
                        messages=messages_list,
                    )
                else:
                    try:
                        return self._collect_completion(
                            proc=proc,
                            model=resolved_model,
                            timeout=effective_timeout,
                            tools=tools,
                            is_worker=True,
                            messages=messages_list,
                        )
                    finally:
                        if self._worker_lock.locked():
                            self._worker_lock.release()
            except Exception:
                self._terminate_worker()
                if self._worker_lock.locked():
                    self._worker_lock.release()
                raise
        else:
            return self._run_oneshot_completion(
                model=resolved_model,
                effort=effort,
                messages=messages_list,
                timeout=effective_timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
            )

