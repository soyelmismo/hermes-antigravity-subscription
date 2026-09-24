"""Antigravity Subscription DirectSDK provider plugin for Hermes Agent."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
)


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
        thinking effort options (low, medium, high) for reasoning-capable models.
        """
        m = (model or "").lower()
        if "gemini-3.1-pro" in m:
            return ("low", "high")
        if "gemini" in m or "flash" in m or "pro" in m:
            return ("low", "medium", "high")
        if "claude" in m:
            return ("low", "medium", "high")
        if "gpt" in m:
            return ()
        return ("low", "medium", "high")

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
        becomes `gemini-3.8-flash`), allowing Hermes' native reasoning effort picker to handle
        the thinking depth cleanly.
        """
        try:
            from .client import resolve_agy_command
        except ImportError:
            # Loaded outside a package (e.g. a flat source tree under test):
            # the absolute name is the same module.
            from client import resolve_agy_command

        cmd = resolve_agy_command()
        try:
            res = subprocess.run(
                [cmd, "models"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw_models: list[str] = []
            for raw_line in res.stdout.strip().splitlines():
                line = raw_line.strip()
                if not line or "fetching" in line.lower():
                    continue
                parts = line.split()
                if parts:
                    model_id = parts[0]
                    if any(c in model_id.lower() for c in ("gemini", "claude", "gpt", "model")):
                        raw_models.append(model_id)

            if raw_models:
                clean_models: list[str] = []
                seen: set[str] = set()
                for m in raw_models:
                    base = m
                    for suffix in ("-high", "-medium", "-low"):
                        if m.endswith(suffix) and (m.startswith("gemini-") or "flash" in m or "pro" in m):
                            base = m[:-len(suffix)]
                            break
                    if base not in seen:
                        seen.add(base)
                        clean_models.append(base)
                return clean_models
        except Exception as exc:
            logger.debug("Antigravity fetch_models failed: %s", exc)

        return list(_FALLBACK_MODELS)


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
    fallback_models=_FALLBACK_MODELS,
    supports_vision=True,
)

register_provider(antigravity_profile)
