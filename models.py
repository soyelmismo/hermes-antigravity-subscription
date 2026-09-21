"""Model catalog mapping and reasoning effort resolution for Antigravity."""

from __future__ import annotations

from typing import Any

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


def resolve_model_and_effort(
    model: str | None,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None]:
    """Map user/hermes model request to concrete CLI model ID and effort level."""
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
        return f"{base_model}-{concrete_effort}", concrete_effort
    elif base_model.startswith("gemini-") and "flash" in base_model:
        concrete_effort = effort if effort in ("low", "medium", "high") else "medium"
        return f"{base_model}-{concrete_effort}", concrete_effort
    elif suffix_effort and effort != suffix_effort:
        # Suffix was in model ID but user requested different reasoning effort
        concrete_effort = effort if effort in ("low", "medium", "high") else "medium"
        return f"{base_model}-{concrete_effort}", concrete_effort
    else:
        return m, effort
