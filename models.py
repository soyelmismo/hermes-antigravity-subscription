"""Model catalog mapping and reasoning effort resolution for Antigravity."""

from __future__ import annotations

import importlib
import logging
import re
import subprocess
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Efforts that `agy --effort` accepts, per base model. A model that agy lists
# as `<base>-low|medium|high` takes --effort; a model it lists by bare name
# does not (agy 1.2.x rejects --effort for those, and rejects a missing
# --effort for the others).
#
# `agy models` is the source of truth. This table only lets known models
# resolve without spawning a process, and covers the case where `agy models`
# fails. An empty tuple means "bare name, no --effort".
_KNOWN_EFFORTS: dict[str, tuple[str, ...]] = {
    # agy 1.5.16 lists -xhigh and accepts `--effort xhigh` for it.
    "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("low", "medium", "high"),
    "gemini-3.1-pro": ("low", "high"),
    "claude-sonnet-4-6": (),
    "claude-opus-4-6-thinking": (),
    "gpt-oss-120b": ("medium",),
}

_FALLBACK_MODELS = list(_KNOWN_EFFORTS)


def fallback_rows() -> list[str]:
    """The built-in fallback catalog, marked as a placeholder when Hermes can say so.

    Hermes' provider-catalog cache treats a plain list as authoritative: it stores it
    under the normal provider TTL and overwrites whatever the account really listed.
    Answering a failed ``agy models`` probe with a plain list therefore pinned the
    ``/model`` picker to these legacy families for hours after one transient failure,
    until somebody ran ``/model --refresh`` — the account's real catalog (with the
    current families) was still on disk but had been replaced by the fallback.

    ``CuratedFallbackModels`` is the marker core understands
    (``hermes_cli/models.py::_live_result_entry``): a marked row never
    replaces a verified one for the same credentials, and when it is stored
    it expires on the short fallback TTL (~60s) instead of the full provider
    TTL. On a core without the marker the plain list keeps the old behaviour.
    """
    marker = _fallback_marker()
    if marker is None:
        return list(_FALLBACK_MODELS)
    return marker(_FALLBACK_MODELS)


def _fallback_marker() -> type | None:
    """The marker class Hermes' catalog cache honours, or None on a core without it.

    It lives in ``hermes_cli.models_catalog_static`` since 2026-10-02 and in
    ``hermes_cli.models`` on the cores between 2026-09-23 and that date.
    """
    for module_name in ("hermes_cli.models_catalog_static", "hermes_cli.models"):
        try:
            marker = getattr(importlib.import_module(module_name), "CuratedFallbackModels", None)
        except Exception:  # a core without that module
            continue
        if isinstance(marker, type):
            return marker
    return None


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
    "claude-sonnet-5-5": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6-thinking",
    "claude-opus": "claude-opus-4-6-thinking",
    "claude-opus-5-5": "claude-opus-4-6-thinking",
}

# Effort suffixes agy appends to a base model id. `-xhigh` is a level of its
# own above `high`: issue #47 took it for a listing duplicate of `-high`, but
# agy 1.5.16 accepts `--effort xhigh` for gemini-4-hydrogen and thinks about
# 75% longer than at `high` on the same prompt. A model agy lists without it
# maps a request for it to `high` (`_nearest_effort`).
_EFFORT_SUFFIXES = (
    ("-xhigh", "xhigh"),
    ("-high", "high"),
    ("-medium", "medium"),
    ("-low", "low"),
)
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}

_SPINNER_CHARS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SKIP_LINE_PREFIXES = ("error:", "usage:", "available models:", "warning:", "--")

# A successful `agy models` result is reused for this long. Short enough that a
# family agy added, renamed or moved to another effort set stops resolving
# through the heuristics within minutes, not an hour.
_CATALOG_TTL_SECONDS = 900.0
_CATALOG_FAILURE_TTL_SECONDS = 60.0
# A model the cached snapshot does not list triggers one re-probe at most this
# often, so a mistyped name still costs no process spawn per request.
_CATALOG_UNKNOWN_REPROBE_SECONDS = 60.0
_catalog_lock = threading.Lock()
# (expires_at, catalog, probed_at) on the monotonic clock.
_catalog_cache: tuple[float, dict[str, tuple[str, ...]], float] | None = None
_catalog_warned_at = float("-inf")


def _normalize_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    e = str(effort).strip().lower()
    if e in ("none", "off", "minimal"):
        return "low"
    if e in ("xhigh", "max", "ultra"):
        # The strongest level agy takes. A model without it gets `high`
        # from `_nearest_effort`, as before.
        return "xhigh"
    if e in ("low", "medium", "high"):
        return e
    return "medium"


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    """Split `gemini-3.8-flash-high` into (`gemini-3.8-flash`, `high`).

    A trailing `-xhigh` splits as the `xhigh` effort, so `gemini-4-hydrogen-xhigh`
    groups into its base model instead of surfacing as a bare-name model.
    """
    for suffix, effort in _EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)], effort
    return model, None


def parse_catalog(text: str) -> dict[str, tuple[str, ...]]:
    """Group `agy models` output as {base model: supported efforts}.

    `gemini-3.8-flash-{low,medium,high}` becomes one `gemini-3.8-flash` entry
    with three efforts; an id with no effort suffix maps to an empty tuple.
    A trailing `-xhigh` groups into its base as the `xhigh` effort, so
    `gemini-4-hydrogen-xhigh` never surfaces as a separate bare-name model.
    Insertion order follows the order agy printed the models in.
    """
    grouped: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "fetching" in line.lower() or line.startswith(_SPINNER_CHARS):
            continue
        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in _SKIP_LINE_PREFIXES):
            continue
        model_id = line.split()[0]
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", model_id):
            continue
        base, effort = split_effort_suffix(model_id)
        efforts = grouped.setdefault(base, [])
        if effort and effort not in efforts:
            efforts.append(effort)
    return {
        base: tuple(sorted(efforts, key=_EFFORT_RANK.__getitem__))
        for base, efforts in grouped.items()
    }


def load_catalog(
    timeout: float = 15.0, *, only_if_stale: bool = False, max_age: float | None = None
) -> dict[str, tuple[str, ...]]:
    """Run `agy models`, cache the parsed result, and return it.

    Returns an empty dict when agy is missing, slow, or prints nothing usable.
    Failures are cached briefly so an absent agy does not cost a process
    spawn per request, and are logged at `warning` once per failure window
    (the traceback stays at `debug`). With `only_if_stale`, a thread that
    waited on the lock reuses the result the previous holder just cached
    instead of spawning agy again; `max_age` makes it re-probe a snapshot
    probed longer ago than that.

    A failed probe keeps the last good snapshot in the cache (under the
    failure TTL), so effort resolution keeps answering from the account's
    real catalog instead of the built-in table while agy is unavailable.
    """
    global _catalog_cache, _catalog_warned_at
    try:
        from .process import _no_window, resolve_agy_command
    except ImportError:
        from process import _no_window, resolve_agy_command

    with _catalog_lock:
        if only_if_stale:
            fresh = _cached_catalog(max_age=max_age)
            if fresh is not None:
                return fresh
        previous = _catalog_cache
        try:
            res = subprocess.run(
                [resolve_agy_command(), "models"],
                capture_output=True,
                text=True,
                timeout=timeout,
                **_no_window(),
            )
            catalog = parse_catalog(res.stdout)
            failure = "" if catalog else f"exit {res.returncode}, no model ids in output"
        except Exception as exc:
            catalog = {}
            failure = str(exc) or type(exc).__name__
            logger.debug("Antigravity model catalog probe failed", exc_info=True)
        now = time.monotonic()
        if catalog:
            _catalog_cache = (now + _CATALOG_TTL_SECONDS, catalog, now)
            return catalog
        if now - _catalog_warned_at >= _CATALOG_FAILURE_TTL_SECONDS:
            _catalog_warned_at = now
            logger.warning(
                "Antigravity model catalog unavailable (%s); serving built-in fallback, "
                "retrying in %ss",
                failure,
                int(_CATALOG_FAILURE_TTL_SECONDS),
            )
        kept = previous[1] if previous is not None and previous[1] else {}
        _catalog_cache = (now + _CATALOG_FAILURE_TTL_SECONDS, kept, now)
        return catalog


def _cached_catalog(*, max_age: float | None = None) -> dict[str, tuple[str, ...]] | None:
    """The cached catalog while it is unexpired and, with `max_age`, recently probed."""
    cache = _catalog_cache
    if cache is None:
        return None
    now = time.monotonic()
    if now >= cache[0]:
        return None
    if max_age is not None and len(cache) > 2 and now - cache[2] >= max_age:
        return None
    return cache[1]


def _unknown_probe_due() -> bool:
    """True when the cached snapshot is older than the unknown-model re-probe interval."""
    cache = _catalog_cache
    if cache is None or len(cache) < 3:
        return False
    return time.monotonic() - cache[2] >= _CATALOG_UNKNOWN_REPROBE_SECONDS


def model_efforts(base_model: str, *, refresh: bool = True) -> tuple[str, ...] | None:
    """Efforts agy accepts for `base_model`; () means bare name, None means unknown.

    Order: a fresh `agy models` result, then the built-in table, then (when
    `refresh` is true) a new `agy models` call. While a cached result is
    fresh, a model it does not list is not looked up again until the cache
    expires, so an unknown name costs no process spawn per request.
    """
    cached = _cached_catalog()
    if cached is not None and base_model in cached:
        return cached[base_model]
    if base_model in _KNOWN_EFFORTS:
        return _KNOWN_EFFORTS[base_model]
    if refresh and cached is None:
        return load_catalog(only_if_stale=True).get(base_model)
    return None


def _heuristic_efforts(base_model: str, has_suffix: bool) -> tuple[str, ...]:
    """Efforts for a model that neither agy nor the built-in table lists."""
    if base_model.startswith("gemini-") and "pro" in base_model:
        return ("low", "high")
    if base_model.startswith("gemini-") and "flash" in base_model:
        return ("low", "medium", "high")
    # A suffixed id is an effort variant; a bare unknown id takes no --effort.
    return ("low", "medium", "high") if has_suffix else ()


def _nearest_effort(requested: str, supported: tuple[str, ...]) -> str:
    """Closest supported effort; on a tie the stronger one wins."""
    target = _EFFORT_RANK[requested]
    return min(
        supported,
        key=lambda e: (abs(_EFFORT_RANK[e] - target), -_EFFORT_RANK[e]),
    )


def resolve_model_alias(
    model: str, catalog: dict[str, tuple[str, ...]] | None = None
) -> str:
    """Resolve user alias to concrete base model, prioritizing the live agy catalog."""
    m = (model or "gemini-3.8-flash").strip()
    lower_m = m.lower()

    if catalog:
        # Exact match or base match already exists in catalog
        if lower_m in catalog:
            return lower_m
        base, suffix = split_effort_suffix(lower_m)
        if base in catalog:
            return lower_m

        lookup_term = "flash" if base == "default" else base

        # Dynamic family matching against whatever models agy actually returned
        family_keywords = {
            ("sonnet", "claude-sonnet"): "sonnet",
            ("opus", "claude-opus"): "opus",
            ("pro", "gemini-pro", "gemini-3.1"): "pro",
            ("flash", "gemini-flash", "gemini-3.8"): "flash",
            ("gpt", "gpt-oss", "oss"): "gpt",
        }
        for triggers, kw in family_keywords.items():
            if lookup_term in triggers:
                matches = [k for k in catalog if kw in k.lower()]
                if matches:
                    # Sort descending so newer versions win if multiple exist (5-5 before 4-6, 3.8 before 3.7)
                    best = sorted(matches, reverse=True)[0]
                    return f"{best}-{suffix}" if suffix else best

        # General substring match against catalog
        matches = [k for k in catalog if lookup_term in k.lower()]
        if matches:
            best = sorted(matches, reverse=True)[0]
            return f"{best}-{suffix}" if suffix else best

        # Check explicit fallback aliases if not matched dynamically in catalog
        if lookup_term in _MODEL_ALIASES:
            aliased = _MODEL_ALIASES[lookup_term]
            if aliased in catalog:
                return f"{aliased}-{suffix}" if suffix else aliased

    base, suffix = split_effort_suffix(lower_m)
    fallback = _MODEL_ALIASES.get(base, base)
    if suffix and fallback != base:
        return f"{fallback}-{suffix}"
    return _MODEL_ALIASES.get(lower_m, m)


def resolve_model_and_effort(
    model: str | None,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None]:
    """Map user/hermes model request to concrete CLI model ID and effort level.

    Returns (model id, effort). The effort is None for models that agy selects
    by bare name; the client then omits --effort.
    """
    raw_model = str(model or "gemini-3.8-flash").strip()
    catalog = _cached_catalog()
    if catalog is None:
        catalog = load_catalog(only_if_stale=True)

    m = resolve_model_alias(raw_model, catalog)

    base_model, suffix_effort = split_effort_suffix(m)
    supported = model_efforts(base_model, refresh=False)
    if supported is None and _unknown_probe_due():
        # The warm snapshot predates this model: agy may have added the family.
        load_catalog(only_if_stale=True, max_age=_CATALOG_UNKNOWN_REPROBE_SECONDS)
        base_model, suffix_effort = split_effort_suffix(
            resolve_model_alias(raw_model, _cached_catalog())
        )
        supported = model_efforts(base_model, refresh=False)
    if supported is None:
        supported = _heuristic_efforts(base_model, suffix_effort is not None)
    if not supported:
        return base_model, None

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

    concrete_effort = _nearest_effort(effort, supported)
    return f"{base_model}-{concrete_effort}", concrete_effort
