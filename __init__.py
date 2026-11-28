"""Antigravity Subscription DirectSDK provider plugin for Hermes Agent."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import fields
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _cached_catalog,
        _heuristic_efforts,
        fallback_rows,
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
        fallback_rows,
        load_catalog,
        model_efforts,
        resolve_model_alias,
        split_effort_suffix,
    )

logger = logging.getLogger(__name__)

# Single-message prompt ceiling measured against agy 1.x (issue #49): the CLI
# silently truncates one Gemini-route user message (first seen at ~75,800
# input tokens; later fillers point at a ~190k-character cut), keeps the head
# and drops the tail, and reports no error. Full sends that do
# not fit one message are split into consecutive turns of one conversation
# (prompt._format_messages_as_segments), which lifts that ceiling.
#
# The conversation itself has a second, larger bound. Measured with agy
# 1.5.15 + gemini-4-neon-low: eleven ~56k-token turns in one stream-json
# process raised no error (no max_trajectory_tokens), but once a turn
# reported a context above ~512k (518,207 after turn 9; 462,145 after turn 8
# did not), the next turn ran agy's checkpointer first: a summary call, after
# which the context dropped from ~592k to ~73k. The agy binary carries
# checkpointer configs with max_token_limit 128000 / 256000 / 512000. Hermes
# must compact before agy summarizes on its own, so Gemini routes declare
# 512,000 minus 32,768 output reserve (the checkpointer's own
# max_output_tokens) minus ~19k margin for the next turn's delta.
#
# With splitting disabled (HERMES_ANTIGRAVITY_SEGMENT_WEIGHT=0) a full send is
# one message again, so the single-message ceiling applies.
_GEMINI_SINGLE_TURN_CONTEXT = 75_000
_GEMINI_CONTEXT_LENGTH = 460_000
_DEFAULT_CONTEXT_LENGTH = 239_000
_CONTEXT_LENGTH_OVERRIDE_VARS = (
    "HERMES_ANTIGRAVITY_CONTEXT_LENGTHS",
    "ANTIGRAVITY_CONTEXT_LENGTHS",
)


def _context_length_overrides() -> dict[str, int]:
    """Parse per-model context overrides from ``*_CONTEXT_LENGTHS`` JSON env vars.

    ``HERMES_ANTIGRAVITY_CONTEXT_LENGTHS`` wins over ``ANTIGRAVITY_CONTEXT_LENGTHS``.
    Values must be positive integers; malformed entries are ignored rather than
    failing the request path.
    """
    for var in _CONTEXT_LENGTH_OVERRIDE_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring malformed %s (not valid JSON)", var)
            continue
        if isinstance(parsed, dict):
            out: dict[str, int] = {}
            for name, val in parsed.items():
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    out[str(name).strip().lower()] = val
            if out:
                return out
    return {}


def resolve_context_length(model: str) -> int:
    """Context length for *model*, per-family where measured.

    Exact-name overrides beat family overrides. Every ``gemini-*`` route,
    effort suffixes included, declares 460k (below agy's ~512k checkpoint)
    while full sends are split into parts, and the 75k single-message
    ceiling when splitting is disabled.
    """
    base, _effort = split_effort_suffix(model.strip().lower())
    overrides = _context_length_overrides()
    for name in (model.strip().lower(), base):
        if name and name in overrides:
            return overrides[name]
    if base.startswith("gemini"):
        if _full_send_splitting_enabled(base):
            return _GEMINI_CONTEXT_LENGTH
        return _GEMINI_SINGLE_TURN_CONTEXT
    return _DEFAULT_CONTEXT_LENGTH


def _full_send_splitting_enabled(model: str) -> bool:
    try:
        from .prompt import full_send_segment_budget
    except ImportError:
        try:
            from prompt import full_send_segment_budget
        except ImportError:
            return False
    return full_send_segment_budget(model) is not None


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

        A failed probe returns `models.fallback_rows()`: the built-in families marked as a
        non-authoritative catalog, so Hermes keeps the account's last verified list instead
        of caching the fallback over it. See `fallback_rows` for the cache contract.
        """
        catalog = load_catalog(timeout=timeout)
        if catalog:
            return list(catalog)
        return fallback_rows()

    def get_model_context_length(self, model: str) -> int | None:
        """Declared context window for Antigravity CLI, per model family.

        Gemini routes declare 460,000 tokens. agy 1.x silently truncates a
        single Gemini user message at ~75,800 input tokens (issue #49), so
        full sends are split into consecutive turns of one conversation; the
        conversation runs agy's own checkpoint summary once a turn reports
        more than ~512k, so Hermes compaction must fire below that. With
        splitting disabled the 75,000 single-message ceiling applies again.
        Claude routes have no observed single-turn cap under the compiled
        trajectory threshold and keep the 239,000 default (239,616 tokens /
        256k - 16k output tokens). Multi-turn trajectory persistence via
        native --conversation prevents hitting the single-turn limit.

        Overrides: ``HERMES_ANTIGRAVITY_CONTEXT_LENGTH`` /
        ``ANTIGRAVITY_CONTEXT_LENGTH`` set one global value;
        ``HERMES_ANTIGRAVITY_CONTEXT_LENGTHS`` /
        ``ANTIGRAVITY_CONTEXT_LENGTHS`` take a JSON object mapping a model
        name or family prefix (e.g. ``"gemini-4-argon"``) to a token count.
        The global override wins over the per-model table.
        """
        env_val = os.environ.get("HERMES_ANTIGRAVITY_CONTEXT_LENGTH") or os.environ.get("ANTIGRAVITY_CONTEXT_LENGTH")
        if env_val:
            try:
                val = int(env_val.strip())
                if val > 0:
                    return val
            except ValueError:
                pass
        return resolve_context_length(model)

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

    def setup_status(self, **kwargs: Any) -> dict[str, Any]:
        """Installed / signed-in / plan / detail from the CLI itself."""
        try:
            try:
                from .setup_hooks import setup_status as _setup_status
            except ImportError:
                from setup_hooks import setup_status as _setup_status
            return _setup_status(**kwargs)
        except Exception as exc:
            logger.debug("Antigravity setup_status failed: %s", exc)
            try:
                try:
                    from .process import resolve_agy_command as _resolve_agy_command
                    from .setup_hooks import _is_executable_command
                except ImportError:
                    from process import resolve_agy_command as _resolve_agy_command
                    from setup_hooks import _is_executable_command
                available = _is_executable_command(_resolve_agy_command())
            except Exception:
                available = False
            return {
                "available": available,
                "logged_in": False,
                "plan": "",
                "detail": str(exc),
                "login_command": None,
            }

    def discover_models(self, **kwargs: Any) -> list[dict[str, str]] | None:
        """The account's live picker as ``[{id, label, note}]``, no inference."""
        try:
            try:
                from .setup_hooks import discover_models as _discover_models
            except ImportError:
                from setup_hooks import discover_models as _discover_models
            return _discover_models(**kwargs)
        except Exception as exc:
            logger.debug("Antigravity discover_models failed: %s", exc)
            return None

    def resolve_aux_model(self, *, vision: bool = False, **kwargs: Any) -> str:
        """Cheapest live model id for auxiliary tasks, or ""."""
        if vision:
            return self.default_vision_model() or ""
        try:
            try:
                from .setup_hooks import resolve_aux_model as _resolve_aux_model
            except ImportError:
                from setup_hooks import resolve_aux_model as _resolve_aux_model
            return _resolve_aux_model()
        except Exception:
            return ""

    def default_vision_model(self) -> str | None:
        """A model on this provider that accepts images natively."""
        try:
            try:
                from .setup_hooks import default_vision_model as _default_vision_model
            except ImportError:
                from setup_hooks import default_vision_model as _default_vision_model
            return _default_vision_model()
        except Exception:
            return None


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
    if getattr(error, "code", None) == "agy_native_tool_protocol":
        return {
            "reason": "incomplete_response",
            "retryable": False,
            "should_compress": False,
            "should_rotate_credential": False,
            "should_fallback": True,
        }
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
    if "exceeded the output token limit" in err_str or "previous response was cut off" in err_str:
        return {
            "reason": "server_error",
            "retryable": True,
            "should_compress": False,
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


# ProviderProfile optional field opt-ins.
_PROFILE_EXTRA_FIELDS: dict[str, Any] = {}


class _LazyModelCapabilities(dict):
    """dict answering exact-model capability lookups without enumerating the catalog.

    The core reads this mapping only via ``.get(model, {})``
    (``agent/models_dev._provider_model_capabilities``), so a dynamic ``get``
    lets the plugin declare per-model vision on demand. Declaring
    ``modalities.input`` here makes ``decide_image_input_mode`` resolve
    ``native`` deterministically — offline, with no models.dev fetch and
    WITHOUT depending on the agy catalog: the family is visible from the
    model id's own prefix (``gemini-*`` / ``claude-*``), so a cold catalog
    cache (first turn after gateway start, TTL expiry, agy unavailable)
    cannot flip the verdict to ``text``. Effort-suffixed ids
    (``gemini-3.8-flash-high``) and future families (``gemini-4-*``,
    ``claude-5-*``) are covered by the same prefix rule. Alias ids are
    resolved as a best-effort second pass. Without this declaration the
    verdict flip-flops between native and text on identical turns depending
    on catalog cache state, which is exactly the multimodality flakiness
    this plugin must not have.
    """

    def get(self, model: Any, default: Any = None) -> Any:
        name = str(model or "").strip()
        if not name:
            return default
        verdict = self._declare(name.lower())
        if verdict is not None:
            return verdict
        # Alias ids ("flash", "opus"): resolve through the live catalog when
        # available and re-check the resolved family. Failure keeps the
        # fail-closed default — the core's own probes decide.
        try:
            resolved = resolve_model_alias(name, _cached_catalog()).lower()
            if resolved != name.lower():
                return self._declare(resolved)
        except Exception:
            pass
        return default

    @staticmethod
    def _declare(base: str) -> dict[str, Any] | None:
        base, _suffix = split_effort_suffix(base)
        if base.startswith("gemini"):
            # Canonical override schema (agent/models_dev._override_to_catalog_shape):
            # ``supports_vision`` is the key the core's vision derivation honors;
            # ``modalities`` is kept for raw readers.
            return {
                "supports_vision": True,
                "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
            }
        if base.startswith("claude"):
            return {
                "supports_vision": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
            }
        # gpt-oss and unknown families: no declaration -> the core's own probes decide.
        return None


_PROFILE_FIELD_NAMES = {f.name for f in fields(ProviderProfile)}
if "supports_streaming" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["supports_streaming"] = True
if "models_dev_id" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["models_dev_id"] = "google"
if "model_capabilities" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["model_capabilities"] = _LazyModelCapabilities()

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
    **_PROFILE_EXTRA_FIELDS,
)

register_provider(antigravity_profile)
