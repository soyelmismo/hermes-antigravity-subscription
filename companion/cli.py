"""CLI surface for the Antigravity companion plugin: `hermes antigravity ...`.

The provider plugin (`antigravity-subscription-directsdk`) is a
``kind: model-provider`` plugin, and Hermes never calls ``register(ctx)`` on that
kind, so it cannot offer a CLI command. This companion plugin is a normal
standalone plugin, so it can. It only reads the same on-disk contract.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

try:
    from . import accounts
except ImportError:  # loaded as a top-level module
    import accounts  # type: ignore[no-redef]


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Wire `hermes antigravity <action>` into the Hermes CLI."""
    subs = subparser.add_subparsers(dest="antigravity_action")

    p_list = subs.add_parser("list", help="List Antigravity accounts with eligibility and quota")
    p_list.add_argument("--fast", action="store_true", help="Skip quota probes (no agy calls)")

    p_run = subs.add_parser("run", help="Run the agy CLI as a stored account (interactive by default)")
    p_run.add_argument("label", help="Account label, or 'host' for the original HOME")
    p_run.add_argument("agy_args", nargs=argparse.REMAINDER, help="Arguments passed through to agy")

    p_usage = subs.add_parser("usage", help="Show remaining quota for one account")
    p_usage.add_argument("label", nargs="?", default=accounts.HOST_LABEL, help="Account label (default: host)")

    p_use = subs.add_parser("use", help="Switch the active account in the registry")
    p_use.add_argument("label", help="Account label, or 'host' to return to the host default")

    p_add = subs.add_parser("add", help="Sign in a new Google account and register it")
    p_add.add_argument("--label", default="", help="Optional label (defaults to the account email)")

    p_mode = subs.add_parser("mode", help="Show or set the rotation mode")
    p_mode.add_argument(
        "mode",
        nargs="?",
        choices=("",) + accounts.ROTATION_MODES,
        help="off | quota | round_robin | fixed (omit to show the current one)",
    )

    p_ignite = subs.add_parser("ignite", help="Show or toggle quota ignition for idle 5h windows")
    p_ignite.add_argument("state", nargs="?", choices=("on", "off", "status"), help="Omit to show the current state")

    subparser.set_defaults(func=antigravity_command)


def antigravity_command(args: argparse.Namespace) -> int:
    action = getattr(args, "antigravity_action", None)
    if not action:
        print("Usage: hermes antigravity {list|run|usage|use|mode|ignite}")
        return 2
    if action == "list":
        return _cmd_list(fast=bool(getattr(args, "fast", False)))
    if action == "run":
        return _cmd_run(str(getattr(args, "label", "")), list(getattr(args, "agy_args", []) or []))
    if action == "usage":
        return _cmd_usage(str(getattr(args, "label", accounts.HOST_LABEL) or accounts.HOST_LABEL))
    if action == "use":
        return _cmd_use(str(getattr(args, "label", "")))
    if action == "add":
        return _cmd_add(str(getattr(args, "label", "") or ""))
    if action == "mode":
        return _cmd_mode(str(getattr(args, "mode", "") or ""))
    if action == "ignite":
        return _cmd_ignite(str(getattr(args, "state", "") or ""))
    print(f"Unknown antigravity action: {action}")
    return 2


def _cmd_list(fast: bool) -> int:
    active = accounts.active_store_name()
    header: list[str] = ["ACCOUNT", "EMAIL", "ELIGIBLE", "GEMINI 5h/wk", "CLAUDE+GPT 5h/wk", "RESET", "HOME"]
    rows: list[list[str]] = [header]
    ineligible: list[str] = []

    for account in accounts.list_accounts():
        is_active = (
            str(account["label"]) == active
            if active
            else bool(account.get("host"))
        )
        home_text = str(account["home"])
        short_home = "~/" + home_text.split("/.agy-accounts/")[-1] if "/.agy-accounts/" in home_text else home_text
        if fast:
            eligible_value = account.get("eligible")
            if eligible_value is False:
                eligible_text = "NO"
                ineligible.append(str(account["label"]))
            elif eligible_value is True:
                eligible_text = "yes"
            else:
                eligible_text = "-"
            gemini = claude = reset_text = "-"
            quota_gemini = quota_claude = "-"
        else:
            eligible, windows, note, resets = accounts.probe(account["home"])
            if eligible is True:
                eligible_text = "yes"
            elif eligible is False:
                eligible_text = "NO"
                ineligible.append(str(account["label"]))
            else:
                eligible_text = f"? ({note})" if note else "?"
            gemini = accounts.window_pair(windows, "gemini")
            claude = accounts.window_pair(windows, "claude_gpt")
            quota_gemini = accounts.window_pair(windows, "gemini")
            quota_claude = accounts.window_pair(windows, "claude_gpt")
            reset_text = accounts.next_reset(resets)
        mark = "*" if is_active else " "
        rows.append(
            [
                f"{mark} {account['label']}",
                str(account["email"] or "-"),
                eligible_text,
                quota_gemini,
                quota_claude,
                reset_text,
                short_home,
            ]
        )

    _print_table(rows)

    # Warn only about accounts that cannot be used; eligible accounts stay quiet.
    if ineligible:
        print(f"\nWARNING: not eligible for Antigravity (skipped by quota rotation): {', '.join(ineligible)}")

    print("\nRun an account:   hermes antigravity run <account>")
    print("Add an account:   hermes antigravity add [--label <name>]")
    print("One-shot:         hermes antigravity run <account> -p \"...\"")
    print("Quota:            hermes antigravity usage [account]")
    print("Switch account:   hermes antigravity use <account>")
    print("Rotation:         hermes antigravity mode [mode]     # off | quota | round_robin | fixed")
    print("Idle wake-up:     hermes antigravity ignite [on|off]")
    return 0


def _unknown_account(label: str) -> int:
    print(f"Unknown account '{label}'. Run `hermes antigravity list` to see accounts.")
    return 2


def _cmd_run(label: str, agy_args: list[str]) -> int:
    account = accounts.resolve_account(label)
    if account is None:
        return _unknown_account(label)
    if agy_args and agy_args[0] == "--":
        agy_args = agy_args[1:]
    return accounts.run_agy(account["home"], agy_args)


def _cmd_usage(label: str) -> int:
    account = accounts.resolve_account(label)
    if account is None:
        return _unknown_account(label)

    eligible, windows, note, resets = accounts.probe(account["home"])
    print(f"Account:  {account['label']}")
    if account["email"]:
        print(f"Email:    {account['email']}")
    print(f"Home:     {account['home']}")

    if eligible is False:
        print("Eligible: NO")
        print(f"\nWARNING: not eligible for Antigravity: {note or 'subscription check failed'}")
        return 1
    if eligible is None:
        print(f"Eligible: unknown ({note or 'could not determine'})")
        return 1

    print("Eligible: yes")
    print(f"  Gemini     5h {accounts.pct(windows.get('gemini_5h'))}   weekly {accounts.pct(windows.get('gemini_weekly'))}")
    print(f"  Claude/GPT 5h {accounts.pct(windows.get('claude_gpt_5h'))}   weekly {accounts.pct(windows.get('claude_gpt_weekly'))}")
    resets_note = accounts.reset_note(resets)
    if resets_note:
        print(f"  {resets_note}")
    return 0


def _cmd_use(label: str) -> int:
    wanted = (label or "").strip()
    if not wanted:
        print("Usage: hermes antigravity use <label>   ('host' returns to the host default)")
        return 2
    if wanted == accounts.HOST_LABEL:
        accounts.set_active_label(None)
        registry = accounts.load_registry()
        current = accounts.active_store_name(registry)
        if current:
            print(f"Active account: {current}")
        else:
            print("Active account: none (host default)")
        return 0
    account = accounts.resolve_account(wanted)
    if account is None:
        return _unknown_account(wanted)
    accounts.set_active_label(str(account["label"]))
    print(f"Active account: {account['label']}")
    return 0


def _cmd_add(label: str) -> int:
    """Register a new account by delegating to the provider's official sign-in flow.

    The sign-in (Google OAuth, token file persistence, eligibility check) lives in
    the provider plugin's `auth_handler`, reached through `hermes auth add`. We only
    forward the call so there is a single implementation, and so credentials are
    never touched by this plugin.
    """
    if not sys.stdin.isatty():
        print("`hermes antigravity add` needs an interactive terminal (it opens the Google sign-in).")
        print("Run it from your terminal, or directly:")
        print("  hermes auth add antigravity-subscription-directsdk --label <name>")
        return 2
    exe = shutil.which("hermes") or "hermes"
    cmd = [exe, "auth", "add", "antigravity-subscription-directsdk"]
    wanted = (label or "").strip()
    if wanted:
        cmd += ["--label", wanted]
    print(f"Signing in a new account{' as ' + wanted if wanted else ''}...")
    return subprocess.call(cmd)


def _cmd_mode(mode: str) -> int:
    wanted = (mode or "").strip().lower()
    if wanted and wanted not in accounts.ROTATION_MODES:
        print(f"Invalid mode '{mode}'. Valid modes: {', '.join(accounts.ROTATION_MODES)}")
        return 2
    if wanted:
        accounts.save_rotation_mode(wanted)
    registry = accounts.load_registry()
    current = str(registry.get("rotation_mode") or "").strip().lower()
    origin = "registry"
    if not current:
        env_raw = os.environ.get("ANTIGRAVITY_ROTATION_MODE", "")
        current, origin = env_raw.strip().lower(), "env"
    if not current:
        current, origin = "off", "default"
    print(f"Rotation mode: {current} ({origin})")
    print(f"Effect: {accounts.describe_rotation_mode(current) or 'unknown mode'}")
    return 0


def _cmd_ignite(state: str) -> int:
    raw = os.environ.get("ANTIGRAVITY_QUOTA_IGNITION", "").strip().lower()
    env_enabled = raw in ("1", "true", "yes", "on")
    registry = accounts.load_registry()
    stored = registry.get("quota_ignition")

    if state and state != "status":
        enabled = state == "on"
        accounts.save_quota_ignition(enabled)
        origin = "registry"
    else:
        if isinstance(stored, bool):
            enabled, origin = stored, "registry"
        else:
            enabled, origin = env_enabled, "env" if raw else "default"
    print(f"Quota ignition: {'on' if enabled else 'off'} ({origin})")
    print(f"Effect: {accounts.ignite_effect(enabled)}")
    return 0


def _print_table(rows: list[list[str]]) -> None:
    if not rows:
        return
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def main(argv: list[str] | None = None) -> int:
    """Direct entry point, useful for local testing without the Hermes CLI."""
    parser = argparse.ArgumentParser(prog="hermes antigravity")
    register_cli(parser)
    return antigravity_command(parser.parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
