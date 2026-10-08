"""Antigravity Subscription DirectSDK provider plugin for Hermes Agent."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _cached_catalog,
        _heuristic_efforts,
        load_catalog,
        model_efforts,
        resolve_model_alias,
        split_effort_suffix,
    )
except ImportError:
    # Loaded outside a package (e.g. a flat source tree under test):
    # the absolute name is the same module.
    from models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _cached_catalog,
        _heuristic_efforts,
        load_catalog,
        model_efforts,
        resolve_model_alias,
        split_effort_suffix,
    )

logger = logging.getLogger(__name__)

class AntigravitySubscriptionDirectSDKProfile(ProviderProfile):
    """Google Antigravity Subscription DirectSDK provider profile."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Create the Antigravity client facade."""
        from .client import AntigravityClient

        return AntigravityClient(**client_kwargs)

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        """Declared reasoning-effort vocabulary for models on this provider.

        Enables Hermes /model picker and /reasoning commands to offer appropriate
        thinking effort options for reasoning-capable models. The levels come from
        `agy models`, so they follow what agy accepts per model: Gemini Flash and
        Claude take low/medium/high, Gemini Pro low/high, gpt-oss medium only.
        """
        name = (model or "").strip()
        catalog = _cached_catalog()
        resolved_name = resolve_model_alias(name, catalog)
        base, suffix_effort = split_effort_suffix(resolved_name)
        # No process spawn here: use the cached catalog and the built-in table.
        efforts = model_efforts(base, refresh=False)
        if efforts is None:
            efforts = _heuristic_efforts(base, suffix_effort is not None)
        return efforts

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract reasoning effort and forward as top-level api_kwargs to AntigravityClient."""
        effort = None
        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                effort = "low"
            else:
                effort = reasoning_config.get("effort")
        top_level: dict[str, Any] = {}
        if effort:
            top_level["reasoning_effort"] = str(effort).strip().lower()
        return {}, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        """Query `agy models` and normalize to clean, deduplicated base models.

        Separates model families from thinking efforts (e.g. `gemini-3.8-flash-{low,medium,high}`
        or `claude-opus-5-5-{low,medium,high}` becomes one entry), allowing Hermes' native
        reasoning effort picker to handle the thinking depth cleanly. The result also warms
        the effort table that `resolve_model_and_effort` reads.
        """
        catalog = load_catalog(timeout=timeout)
        if catalog:
            return list(catalog)
        return list(_FALLBACK_MODELS)

    def get_model_context_length(self, model: str) -> int | None:
        """Declared context window for Antigravity CLI.

        Defaults to 239,000 tokens so Hermes compaction triggers safely below agy's
        compiled internal trajectory checkpoint threshold (239,616 tokens / 256k - 16k output tokens),
        maximizing usable context while ensuring Hermes retains sovereign context control. Multi-turn
        trajectory persistence via native --conversation prevents hitting agy's single-turn 100,000 token limit.
        Configurable via HERMES_ANTIGRAVITY_CONTEXT_LENGTH or ANTIGRAVITY_CONTEXT_LENGTH.
        """
        env_val = os.environ.get("HERMES_ANTIGRAVITY_CONTEXT_LENGTH") or os.environ.get("ANTIGRAVITY_CONTEXT_LENGTH")
        if env_val:
            try:
                val = int(env_val.strip())
                if val > 0:
                    return val
            except ValueError:
                pass
        return 239_000

    def classify_api_error(
        self,
        error: Exception,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        message: str = "",
        body: Any = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        return _classify_antigravity_error(
            error,
            status_code=status_code,
            error_code=error_code,
            message=message,
            body=body,
            model=model,
        )

    def fetch_account_usage(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Fetch subscription quota snapshot for /usage, status bar, and desktop."""
        try:
            from .usage import get_account_usage_snapshot
        except ImportError:
            from usage import get_account_usage_snapshot

        return get_account_usage_snapshot(force_refresh=force_refresh)


def _classify_antigravity_error(
    error: Exception,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str = "",
    body: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Classify agy CLI specific runtime errors so Hermes' smart failover /
    recovery pipeline triggers auto-compression, billing failover, or permanent auth abort.
    """
    err_str = f"{error} {message}".lower()
    if any(
        pattern in err_str
        for pattern in (
            "subscriber fell behind updates",
            "stalled for 5s",
            "empty result (status='success')",
            "empty result (status=\"success\")",
            "max_trajectory_tokens",
            "max trajectory tokens",
        )
    ):
        return {
            "reason": "context_overflow",
            "retryable": True,
            "should_compress": True,
        }
    if any(
        pattern in err_str
        for pattern in (
            "ai credits balance is too low",
            "credits balance is too low",
            "daily or billing cap",
            "billing cap",
        )
    ):
        return {
            "reason": "billing",
            "retryable": False,
            "should_fallback": True,
            "should_compress": False,
        }
    if any(
        pattern in err_str
        for pattern in (
            "verify your account to continue",
            "appeal a terms of service block",
            "terms of service block",
            "terms of service violation",
        )
    ):
        return {
            "reason": "auth_permanent",
            "retryable": False,
            "should_fallback": False,
            "should_compress": False,
        }
    return None


antigravity_profile = AntigravitySubscriptionDirectSDKProfile(
    name="antigravity-subscription-directsdk",
    aliases=("antigravity", "agy", "antigravity-directsdk"),
    display_name="Antigravity Subscription DirectSDK",
    description="Use your Antigravity / Gemini subscription via the official agy CLI",
    base_url="agy://local",
    api_mode="chat_completions",
    auth_type="external_process",
    process_command="agy",
    process_args=("--output-format", "stream-json", "--disable-slash-commands"),
    process_command_env_vars=("ANTIGRAVITY_COMMAND", "AGY_CLI_PATH", "ANTIGRAVITY_CLI_PATH"),
    process_args_env_var="ANTIGRAVITY_ARGS",
    default_aux_model="gemini-3.8-flash",
    fallback_models=(),  # Account-scoped models: discovered dynamically via fetch_models() (agy models)
    supports_vision=True,
    unsupported_response_formats=("json_schema",),
    classify_api_error=_classify_antigravity_error,
)

register_provider(antigravity_profile)
