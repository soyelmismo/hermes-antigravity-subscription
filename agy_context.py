"""Single-frame UTF-8 byte-cap accounting for the ``agy`` CLI.

``agy`` caps a single prompt frame at a measured UTF-8 byte limit (~191,583
bytes on ``agy`` 1.5.15..1.5.18) rather than a fixed token count. When a single
frame exceeds that limit (for example an unsegmented full send, a large
system prompt + tool schema, or an oversized tool-result delta), ``agy`` drops
the tail of the input and reports the cut in its reply or transcript
(``N bytes truncated`` or ``<truncated N bytes>``).

This module provides:
- :func:`prompt_byte_budget`: the byte budget for one prompt frame (default 94%
  of the measured 191,583-byte cap; override via ``HERMES_ANTIGRAVITY_PROMPT_BYTES``
  or ``ANTIGRAVITY_PROMPT_BYTES``, set ``0`` to disable).
- :func:`note_truncation`: detects truncation notices in ``agy`` replies and
  learns a stricter surviving byte cap for subsequent frames.
- :func:`omitted_notice` and :func:`hard_trim`: explicit omission notices and
  head/tail preservation when even the strictest shrink pass exceeds the cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Largest single prompt frame observed to survive agy 1.5.15..1.5.18 unchanged.
MEASURED_SINGLE_TURN_BYTES = 191_583

#: Lower bound below which a learned cap is rejected as implausible.
MIN_LEARNED_SINGLE_TURN_BYTES = 65_536

#: Safety margin below the measured cap to leave room for framing/tokenization.
BUDGET_FRACTION = 0.94

_BUDGET_ENV_VARS = ("HERMES_ANTIGRAVITY_PROMPT_BYTES", "ANTIGRAVITY_PROMPT_BYTES")

# Two truncation notice formats observed from agy:
#   "Your message was truncated after `...` (`228,961 bytes` truncated)"
#   "<truncated 296125 bytes>"
_TRUNCATION_PATTERNS = (
    re.compile(r"your message was truncated[^\n]*?`(\d[\d,]*)\s*bytes`\s*truncated", re.IGNORECASE),
    re.compile(r"<truncated\s+(\d[\d,]*)\s*bytes>", re.IGNORECASE),
)
_TRUNCATION_HINT = re.compile(r"your message was truncated|<truncated\s+\d[\d,]*\s*bytes>", re.IGNORECASE)

_learned_cap: Optional[int] = None
_learned_lock = threading.Lock()


@dataclass(frozen=True)
class TruncationNotice:
    """What an agy reply says was cut from the prompt."""

    dropped_bytes: int
    observed_cap: Optional[int] = None


def _nonneg_int(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "").replace("_", "")
        if text.isdigit():
            return int(text)
    return None


def _positive_int(raw: Any) -> Optional[int]:
    val = _nonneg_int(raw)
    return val if val is not None and val > 0 else None


def _env_cap(names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            continue
        val = _nonneg_int(raw)
        if val is not None:
            return val
    return None


def learned_single_turn_bytes() -> Optional[int]:
    """Cap observed from a real truncation notice, if one has been seen."""
    with _learned_lock:
        return _learned_cap


def single_turn_byte_cap() -> Optional[int]:
    """Byte cap to size prompts against, or None when explicitly disabled (`0`)."""
    configured = _env_cap(_BUDGET_ENV_VARS)
    if configured is not None:
        return None if configured == 0 else configured
    observed = learned_single_turn_bytes()
    if observed is not None:
        return min(MEASURED_SINGLE_TURN_BYTES, observed)
    return MEASURED_SINGLE_TURN_BYTES


def prompt_byte_budget() -> Optional[int]:
    """Byte budget for one assembled prompt frame, or None when disabled (`0`)."""
    cap = single_turn_byte_cap()
    if cap is None:
        return None
    budget = int(cap * BUDGET_FRACTION)
    # Never let a tiny positive env value produce a zero-byte budget.
    return max(budget, 4_096)


def prompt_bytes(text: str) -> int:
    """UTF-8 byte length of a prompt frame, which is what the transport counts."""
    return len(text.encode("utf-8", "replace"))


def parse_truncation_notice(text: str) -> Optional[TruncationNotice]:
    """Parse agy's truncation notice out of a reply, if it carries one."""
    if not text or not isinstance(text, str):
        return None
    if not _TRUNCATION_HINT.search(text):
        return None
    for pattern in _TRUNCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            dropped = _positive_int(match.group(1))
            if dropped is not None:
                return TruncationNotice(dropped_bytes=dropped)
    return TruncationNotice(dropped_bytes=0)


def note_truncation(
    text: str,
    prompt_size: int,
    *,
    learn: bool = True,
) -> Optional[TruncationNotice]:
    """Record a truncation notice and optionally tighten the learned frame cap.

    Only trusted transport metadata should pass ``learn=True``; untrusted model
    reply text should pass ``learn=False`` so echoed logs or prompt injections
    cannot lower the process-global cap. Even when ``learn=True``, implausibly
    small surviving values (below :data:`MIN_LEARNED_SINGLE_TURN_BYTES`) are
    ignored.
    """
    notice = parse_truncation_notice(text)
    if notice is None:
        return None
    observed: Optional[int] = None
    if notice.dropped_bytes and prompt_size >= MIN_LEARNED_SINGLE_TURN_BYTES:
        surviving = prompt_size - notice.dropped_bytes
        if MIN_LEARNED_SINGLE_TURN_BYTES <= surviving < MEASURED_SINGLE_TURN_BYTES:
            observed = surviving
            if learn:
                global _learned_cap
                with _learned_lock:
                    if _learned_cap is None or surviving < _learned_cap:
                        _learned_cap = surviving
    logger.warning(
        "agy truncated the prompt: %s bytes dropped from a %s-byte frame "
        "(observed cap %s, learned=%s, budget now %s bytes).",
        notice.dropped_bytes or "unknown",
        prompt_size or "unknown",
        observed if observed is not None else "unknown",
        learn and observed is not None,
        prompt_byte_budget(),
    )
    return TruncationNotice(dropped_bytes=notice.dropped_bytes, observed_cap=observed)


def reset_learned_cap() -> None:
    """Forget a learned cap (tests, or after an agy upgrade)."""
    global _learned_cap
    with _learned_lock:
        _learned_cap = None


def omitted_notice(omitted_messages: int, budget: int) -> str:
    """Notice telling the model older turns were omitted to fit the byte cap."""
    plural = "s" if omitted_messages != 1 else ""
    return (
        "### EARLIER CONTEXT OMITTED\n"
        f"The transport accepts about {budget:,} bytes per prompt, so {omitted_messages} older "
        f"message{plural} were omitted from this transcript. Ask the user for specifics, or "
        "re-read a file, if the omitted turn matters."
    )


def _utf8_prefix(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8", "replace")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", "ignore")


def _utf8_suffix(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8", "replace")
    if len(raw) <= max_bytes:
        return text
    return raw[-max_bytes:].decode("utf-8", "ignore")


def hard_trim(text: str, budget: int) -> str:
    """Last-resort cut preserving head (system/tools) and tail (live request)."""
    total_bytes = prompt_bytes(text)
    if total_bytes <= budget:
        return text
    marker_tpl = "\n\n[... {} bytes of transcript omitted to fit the agy frame ...]\n\n"
    sample_marker = marker_tpl.format(total_bytes)
    marker_reserve = prompt_bytes(sample_marker) + 16
    room = max(budget - marker_reserve, 0)
    head_budget = room // 2
    tail_budget = room - head_budget
    head = _utf8_prefix(text, head_budget)
    tail = _utf8_suffix(text, tail_budget)
    dropped = max(total_bytes - prompt_bytes(head) - prompt_bytes(tail), 0)
    marker = marker_tpl.format(dropped)
    trimmed = head + marker + tail
    if prompt_bytes(trimmed) > budget:
        trimmed = _utf8_prefix(trimmed, budget)
    return trimmed


def describe_budget() -> dict[str, Any]:
    """Diagnostic snapshot for logs and tests."""
    return {
        "measured_cap": MEASURED_SINGLE_TURN_BYTES,
        "learned_cap": learned_single_turn_bytes(),
        "effective_cap": single_turn_byte_cap(),
        "budget": prompt_byte_budget(),
    }


def format_describe() -> str:
    return json.dumps(describe_budget(), sort_keys=True)
