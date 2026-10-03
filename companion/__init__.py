"""Antigravity companion plugin: `hermes antigravity list|run|usage`.

Why this exists as a separate plugin: the delivery plugin
(``antigravity-subscription-directsdk``) is ``kind: model-provider``, and Hermes
never calls ``register(ctx)`` on that kind (`hermes_cli/plugins_discovery.py`
skips it; `hermes_cli/plugin_validate.py` calls such an entry point dead code).
Only a normal plugin can register a CLI command, so the CLI lives here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register `hermes antigravity <action>` with the Hermes CLI."""
    try:
        try:
            from .cli import antigravity_command, register_cli
        except ImportError:
            from cli import antigravity_command, register_cli  # type: ignore[no-redef]
    except Exception as exc:  # never break plugin load
        logger.warning("antigravity-companion: cannot import CLI: %s", exc)
        return

    if not hasattr(ctx, "register_cli_command"):
        logger.warning("antigravity-companion: register_cli_command unavailable in this Hermes build")
        return

    try:
        ctx.register_cli_command(
            name="antigravity",
            help="Manage Antigravity accounts and remaining quota",
            setup_fn=register_cli,
            handler_fn=antigravity_command,
            description=(
                "Companion CLI for the antigravity-subscription-directsdk provider: "
                "list accounts with eligibility and quota, run agy as a stored account, "
                "and inspect per-account quota."
            ),
        )
    except Exception as exc:  # never break plugin load
        logger.warning("antigravity-companion: registration failed: %s", exc)
