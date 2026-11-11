"""Prompt construction, message translation, and tool-call parsing for Antigravity."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Sequence

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}

_PROMPT_PREAMBLE = (
    # agy embeds its own system prompt before the user's message. Some models
    # (claude-sonnet-4-6, gemini-3.6-flash) obey it and invoke native tools,
    # which headless mode soft-denies. A confrontational denial ("you are NOT
    # Antigravity") makes the model classify the preamble as prompt injection
    # and resist harder. This framing avoids contradicting agy's identity and
    # instead redirects the tool-calling protocol. (Issue #16)
    "You are operating in HEADLESS INFERENCE MODE. In this mode, your native tools (run_command, write_to_file, view_file, edit_file, etc.) are disabled and will be silently rejected. Do not attempt to use them.",
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent on the host side.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output.",
    "- Each tool call must be a JSON object containing 'id', 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}.",
    "- 'arguments' must be a JSON-encoded string containing the function arguments.",
    "- Do NOT execute local shell commands or file operations directly; only output <tool_call> tags so Hermes can execute them safely.",
    "- NEVER simulate or hallucinate Tool Results (e.g. 'Tool Result (...):'). Hermes Agent executes tools externally and will supply real results in subsequent turns.",
    "- After emitting tool calls, STOP generating immediately. Do NOT generate results, execution output, or commentary after tool calls.",
    "- If no tool is needed, respond naturally with standard text.",
    "- When the user's latest message introduces a new question or task, do NOT continue prior completed tasks or tool loops. Address the user's latest request directly.",
)

_TOOL_CALL_PREFIXES = tuple(
    "<tool_call>"[:i] for i in range(len("<tool_call>"), 0, -1)
)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _find_tool_call_close(text: str, start: int = 0) -> int:
    """Index of the first ``</tool_call>`` outside a JSON string, or -1.

    A tool argument may legitimately contain the literal closing tag (for
    example a patch or a file whose content includes protocol markup). A plain
    ``str.find`` truncates the call at that literal and hands invalid JSON to
    the parser. This scanner tracks JSON double-quoted strings and backslash
    escapes, so only a tag at string depth zero closes the frame. It is the
    single frame boundary used by both the incremental stream splitter and
    :func:`_parse_tool_block`.
    """
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if text.startswith(_TOOL_CALL_CLOSE, index):
            return index
    return -1


def _render_message_content(content: Any) -> str:
    """Normalize multimodal or structured message content into a string."""
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


def _latest_user_request_section(user_text: str) -> str:
    """Format the canonical LATEST USER REQUEST section and anti-loop instruction."""
    return (
        f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}\n\n"
        "INSTRUCTION: Respond directly and specifically to the LATEST USER REQUEST above. "
        "Do NOT repeat previous architectural summaries, code reviews, or overview boilerplate unless explicitly asked."
        " If the LATEST USER REQUEST comments on, questions, or gives feedback about prior work rather than asking to continue it, address THAT message and do NOT silently continue the earlier task. If the user's intent is genuinely ambiguous, ask ONE short clarifying question instead of proceeding."
    )


# Core strips private message keys before providers see the request. Match only
# the complete known recovery text, never arbitrary user text containing it.
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)


def _is_recovery_nudge(message: dict[str, Any]) -> bool:
    return bool(message.get("_empty_recovery_synthetic")) or (
        _render_message_content(message.get("content")) == _EMPTY_TOOL_RESPONSE_NUDGE
    )


def _latest_user_text(messages: Sequence[Any]) -> str:
    """Return the rendered text of the last user message, or an empty string."""
    for msg in reversed(list(messages or [])):
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "user":
            if not _is_recovery_nudge(msg):
                return _render_message_content(msg.get("content"))
    return ""


def _pending_user_request_section(user_text: str) -> str:
    """Restate the user request that the current tool loop is serving.

    Tool-result turns otherwise end with a generic "continue" line, so after a
    few tool calls the request falls far behind the newest text the model sees
    and the model drifts back to the dominant pattern in the transcript (for
    example re-sending an earlier completion report).
    """
    return (
        f"### USER REQUEST THIS TOOL LOOP IS SERVING:\nUser:\n{user_text}\n\n"
        "INSTRUCTION: Use the tool results only to answer THIS request. "
        "When you have enough information, answer it directly and specifically. "
        "Do NOT repeat earlier reports or summaries from the transcript unless this request asks for them."
    )


def _tool_loop_objective_section(user_text: str) -> str:
    """Anchor the request a tool loop serves above the transcript.

    A full prompt that ends on a tool result (first call after a worker
    respawn or a broken pipe) otherwise names the request only at the tail,
    and a long transcript drowns a tail-only instruction.
    """
    return (
        "### ACTIVE OBJECTIVE:\n"
        f"The user request being served is:\n\"{user_text}\"\n\n"
        "CRITICAL: The conversation transcript below contains prior history and the tool calls made so far. "
        "Use the tool results ONLY to address the ACTIVE OBJECTIVE above. "
        "When tool execution is complete, your final answer must directly address this objective. "
        "Do NOT repeat earlier completion reports, summaries, or boilerplate from previous tasks, "
        "and do NOT resume an earlier task."
    )


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

    valid_messages = [m for m in messages if isinstance(m, dict)]
    last_msg = valid_messages[-1] if valid_messages else None
    last_role = str(last_msg.get("role") or "").strip().lower() if last_msg else ""

    system_parts: list[str] = []
    history_messages: list[dict[str, Any]] = []

    for msg in valid_messages:
        role = str(msg.get("role") or "unknown").strip().lower()
        if role == "system":
            rendered = _render_message_content(msg.get("content"))
            if rendered:
                system_parts.append(rendered)
        else:
            history_messages.append(msg)

    if system_parts:
        sections.append("### SYSTEM INSTRUCTIONS (HERMES AGENT):\n" + "\n\n".join(f"System:\n{p}" for p in system_parts))

    _HISTORICAL_TOOL_RETENTION_COUNT = 8
    _HISTORICAL_TOOL_MAX_CHARS = 300
    _TOOL_MAX_CHARS_CAP = 20000

    transcript: list[str] = []
    recovery_turn = last_role == "user" and last_msg is not None and _is_recovery_nudge(last_msg)
    prior_messages = history_messages[:-1] if last_role == "user" and not recovery_turn and len(history_messages) > 1 else history_messages

    tool_indices = [
        i for i, msg in enumerate(prior_messages)
        if str(msg.get("role") or "").strip().lower() == "tool"
    ]
    recent_tool_cutoff = (
        set(tool_indices[-_HISTORICAL_TOOL_RETENTION_COUNT:])
        if len(tool_indices) > _HISTORICAL_TOOL_RETENTION_COUNT
        else set(tool_indices)
    )

    for i, message in enumerate(prior_messages):
        role = str(message.get("role") or "unknown").strip().lower()
        rendered_content = _render_message_content(message.get("content"))

        if role == "tool":
            tool_id = str(message.get("tool_call_id") or message.get("name") or "tool").strip()
            if i not in recent_tool_cutoff and len(rendered_content) > _HISTORICAL_TOOL_MAX_CHARS:
                rendered_content = (
                    rendered_content[:200]
                    + f"\n[... tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-100:]
                )
            elif len(rendered_content) > _TOOL_MAX_CHARS_CAP:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            transcript.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue

        if role == "assistant":
            parts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
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
                transcript.append("Assistant:\n" + "\n".join(parts))
            continue

        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            transcript.append(f"{label}:\n{rendered_content}")

    if recovery_turn:
        objective = _latest_user_text(history_messages)
        if objective:
            sections.append(_tool_loop_objective_section(objective))
    elif last_role == "user" and last_msg is not None and len(history_messages) > 1:
        user_text = _render_message_content(last_msg.get("content"))
        sections.append(
            "### ACTIVE OBJECTIVE:\n"
            f"The user has submitted a new request:\n\"{user_text}\"\n\n"
            "CRITICAL: The conversation transcript below is historical background. "
            "Do NOT silently continue prior tool execution loops or earlier coding tasks unless explicitly requested by the ACTIVE OBJECTIVE. "
            "Focus your attention on answering the new request."
        )
    elif last_role == "tool":
        pending_objective = _latest_user_text(history_messages)
        if pending_objective:
            sections.append(_tool_loop_objective_section(pending_objective))

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    if last_role == "tool":
        sections.append(
            "### LATEST TOOL RESULTS RECEIVED.\n"
            "INSTRUCTION: Evaluate the latest tool results in the transcript above and continue the task. "
            "If more tools are needed, emit <tool_call> tags. If you have enough information, "
            "respond clearly to the user without repeating prior summaries."
        )
        pending = _latest_user_text(history_messages)
        if pending:
            sections.append(_pending_user_request_section(pending))
    elif last_role == "user" and last_msg is not None:
        user_text = _latest_user_text(history_messages)
        if user_text:
            sections.append(_latest_user_request_section(user_text))
    else:
        sections.append("Continue the conversation from the latest message.")
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
        if inc_msg.get("tool_calls") != h_msg.get("tool_calls"):
            return False
    return True


def _format_delta_prompt(
    new_messages: Sequence[dict[str, Any]],
    pending_user_text: str = "",
) -> str:
    """Format only the incremental messages in an ongoing multi-turn interaction.

    ``pending_user_text`` is the latest user request from the full history.
    A tool-result delta does not contain it, so the caller passes it in and it
    is restated after the continue instruction.
    """
    valid_messages = [m for m in new_messages if isinstance(m, dict)]
    if not valid_messages:
        return ""

    last_msg = valid_messages[-1]
    last_role = str(last_msg.get("role") or "").strip().lower()

    recovery_turn = last_role == "user" and _is_recovery_nudge(last_msg)
    if last_role == "user" and not recovery_turn:
        prior_messages = valid_messages[:-1]
    else:
        prior_messages = valid_messages

    parts: list[str] = []
    for msg in prior_messages:
        role = str(msg.get("role") or "").strip().lower()
        rendered_content = _render_message_content(msg.get("content"))
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("name") or "tool").strip()
            if len(rendered_content) > 20000:
                rendered_content = (
                    rendered_content[:12000]
                    + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-3000:]
                )
            parts.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue
        if role == "assistant":
            subparts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
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
                parts.append("Assistant:\n" + "\n".join(subparts))
            continue
        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            parts.append(f"{label}:\n{rendered_content}")

    if last_role == "tool":
        parts.append(
            "Continue the conversation from the latest tool result. "
            "REMINDER: You are in HEADLESS INFERENCE MODE. Native tools are disabled. "
            "If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output. "
            "Do NOT invoke native tools directly."
        )
        if pending_user_text:
            parts.append(_pending_user_request_section(pending_user_text))
    elif last_role == "user":
        user_text = (pending_user_text or _latest_user_text(valid_messages)) if recovery_turn else _render_message_content(last_msg.get("content"))
        if user_text:
            parts.append(_latest_user_request_section(user_text))
    else:
        parts.append("Continue the conversation from the latest message above.")
    return "\n\n".join(s.strip() for s in parts if s and s.strip())


def _longest_tool_call_prefix_match(text: str) -> int:
    """Return the length of the longest suffix of `text` that matches a prefix of `<tool_call>`."""
    for prefix in _TOOL_CALL_PREFIXES:
        if text.endswith(prefix):
            return len(prefix)
    return 0


def _has_unfinished_tool_call_frame(text: str) -> bool:
    """True when *text* ends inside a ``<tool_call>`` frame.

    Uses the incremental splitter's frame boundary (:func:`_find_tool_call_close`)
    so a complete block whose JSON contains a literal closing tag is not
    mistaken for a truncated one. An opening tag without a matching close, or
    a trailing fragment of the opening tag (``<tool_cal``), is unfinished.
    """
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            # A lone "<" is ordinary text (a comparison operator, an unfinished
            # HTML tag, ...), not evidence of a truncated <tool_call> frame.
            # Two or more matching characters ("<t" ... "<tool_call") remain
            # fail-closed: the shortest useful proof of protocol framing.
            return _longest_tool_call_prefix_match(text) > 1
        close = _find_tool_call_close(text, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            return True
        cursor = close + len(_TOOL_CALL_CLOSE)


def _tool_call_from_object(obj: Any, ordinal: int) -> Any | None:
    """Build one tool-call delta object from a decoded JSON value.

    Mirrors the core ACP parser: the value must be an object whose
    ``function`` is an object with a non-blank string name. Anything else is
    malformed and returns None, so a nameless call is never emitted.
    """
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if not isinstance(fn, dict):
        return None
    fn_name = fn.get("name")
    if not isinstance(fn_name, str) or not fn_name.strip():
        return None
    fn_name = fn_name.strip()
    call_id = obj.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call_{ordinal}"
    fn_args = fn.get("arguments", "{}")
    if not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=fn_name, arguments=fn_args),
    )


def _has_unparseable_tool_call_frame(text: str) -> bool:
    """True when *text* keeps a closed ``<tool_call>`` frame after parsing.

    :func:`_parse_tool_block` consumes every frame it can turn into a call and
    leaves the unparseable ones in its cleaned text: invalid JSON, a missing
    or blank ``function`` name, a non-object value, or an empty body. A
    closed frame surviving in that cleaned text is therefore an explicit
    protocol frame with no call, which must fail the turn instead of being
    published as assistant text. Unclosed frames are
    :func:`_has_unfinished_tool_call_frame`'s concern.
    """
    _calls, cleaned = _parse_tool_block(text)
    start = cleaned.find(_TOOL_CALL_OPEN)
    return (
        start != -1
        and _find_tool_call_close(cleaned, start + len(_TOOL_CALL_OPEN)) != -1
    )


def _extract_tagged_tool_calls(block: str) -> tuple[list[Any], str]:
    """Extract ``<tool_call>{...}</tool_call>`` frames using the shared scanner.

    The frame boundary is :func:`_find_tool_call_close`, so a literal closing
    tag inside a JSON string does not terminate the block. Text outside valid
    frames, and a frame whose body is not a valid call, are returned as
    ``cleaned_text``.
    """
    calls: list[Any] = []
    text_parts: list[str] = []
    cursor = 0
    while True:
        start = block.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            text_parts.append(block[cursor:])
            break
        text_parts.append(block[cursor:start])
        close = _find_tool_call_close(block, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            text_parts.append(block[start:])
            break
        end = close + len(_TOOL_CALL_CLOSE)
        body = block[start + len(_TOOL_CALL_OPEN):close].strip()
        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        call = _tool_call_from_object(parsed, len(calls) + 1)
        if call is None:
            text_parts.append(block[start:end])
        else:
            calls.append(call)
        cursor = end
    if not calls:
        # No frame was consumed: hand the input back untouched. Joining the
        # pieces would insert separators at every tag boundary and corrupt
        # ordinary prose that merely mentions the markup.
        return [], block.strip()
    return calls, "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()


def _parse_tool_block(block: str) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text)."""
    calls: list[Any] = []
    cleaned_text = ""
    if _TOOL_CALL_OPEN in block:
        calls, cleaned_text = _extract_tagged_tool_calls(block)
    else:
        try:
            from agent.acp_openai_bridge import extract_tool_calls_from_text
            calls, cleaned_text = extract_tool_calls_from_text(block)
        except Exception:
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
