"""Antigravity Subscription DirectSDK client for Hermes Agent.

Drives the official Antigravity CLI (`agy`) as an external process in
stream-json mode to provide request-scoped completions using the user's
existing Antigravity / Gemini subscription quota.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from .process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        is_authenticated,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        terminate_process,
    )
    from .prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
    )
    from .stream import AntigravityStream, collect_stream_completion
except ImportError:
    from models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        is_authenticated,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        terminate_process,
    )
    from prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
    )
    from stream import AntigravityStream, collect_stream_completion

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300.0

__all__ = [
    "AGY_MARKER_BASE_URL",
    "AntigravityClient",
    "AntigravityStream",
    "_FALLBACK_MODELS",
    "_MODEL_ALIASES",
    "_PROMPT_PREAMBLE",
    "_ROLE_LABELS",
    "_format_delta_prompt",
    "_format_messages_as_prompt",
    "_kill_process_tree",
    "_longest_tool_call_prefix_match",
    "_messages_match_prefix",
    "_normalize_effort",
    "_own_process_group",
    "_parse_tool_block",
    "_render_message_content",
    "is_authenticated",
    "resolve_agy_command",
    "resolve_model_and_effort",
]


class AntigravityClient:
    """OpenAI-compatible client facade driving Antigravity CLI."""

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
        self._isolated_home, self._isolated_gemini_dir = setup_isolated_home(self._cwd)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        self._active_processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._worker_proc: subprocess.Popen | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_history: list[dict[str, Any]] = []
        # Cumulative usage snapshot of the current worker session. agy 1.2.10+
        # persistent workers report cumulative session usage, so per-turn
        # deltas need this baseline. The reference is replaced on every spawn
        # and termination; the dict contents are advanced in place by the
        # owning session's stream. Each stream captures the dict of the
        # session it was created for, so a stream left over from a terminated
        # session cannot corrupt a later session's baseline.
        self._worker_usage_baseline: dict[str, int] = {}
        self._worker_lock = threading.Lock()

    @staticmethod
    def _resolve_real_token_path() -> Path | None:
        return resolve_real_token_path()

    def _resolve_model_and_effort(
        self,
        model: str | None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        return resolve_model_and_effort(model, reasoning_effort)

    def _child_env(self) -> dict[str, str]:
        return build_child_env(self._isolated_home)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        terminate_process(proc)

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
        # The session is gone: any later usage from it belongs to a dead
        # session, and the next spawn must start from a clean baseline.
        self._worker_usage_baseline = {}

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
            self._worker_usage_baseline = {}
            self._active_processes.add(proc)
            return proc

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

        stream_iter = AntigravityStream(
            proc=proc,
            client=self,
            model=model,
            timeout=timeout,
            tools=tools,
            is_worker=False,
        )
        if stream:
            return stream_iter
        return collect_stream_completion(stream_iter)

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
        effective_timeout = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout > 0
            else _DEFAULT_TIMEOUT_SECONDS
        )

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

                with self._lock:
                    worker_usage_baseline = self._worker_usage_baseline
                stream_iter = AntigravityStream(
                    proc=proc,
                    client=self,
                    model=resolved_model,
                    timeout=effective_timeout,
                    tools=tools,
                    is_worker=True,
                    worker_lock=self._worker_lock,
                    messages=messages_list,
                    usage_baseline=worker_usage_baseline,
                )
                if stream:
                    return stream_iter
                try:
                    return collect_stream_completion(stream_iter)
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
