"""Antigravity Subscription DirectSDK provider plugin for Hermes Agent."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .accounts import (
        DEFAULT_ACCOUNTS_DIR,
        add_account,
        check_account_eligibility,
        extract_email_from_token_file,
        fetch_usage_for_home,
        find_account_token_path,
        get_accounts_dir,
        get_active,
        get_rotation_mode,
        list_accounts,
        remove_account,
        sanitize_folder_name,
        set_active,
        set_cooldown,
    )
    from .process import resolve_agy_command
except ImportError:
    from accounts import (
        DEFAULT_ACCOUNTS_DIR,
        add_account,
        check_account_eligibility,
        extract_email_from_token_file,
        fetch_usage_for_home,
        find_account_token_path,
        get_accounts_dir,
        get_active,
        get_rotation_mode,
        list_accounts,
        remove_account,
        sanitize_folder_name,
        set_active,
        set_cooldown,
    )
    from process import resolve_agy_command

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# An interactive sign-in that never returns must not hang the CLI forever.
_LOGIN_TIMEOUT_SECONDS = 600.0

_FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
)


def _rotation_accounts() -> list[dict[str, Any]]:
    """Registered accounts that can serve a request (enabled and eligible)."""
    try:
        return [
            a for a in list_accounts()
            if isinstance(a, dict) and a.get("enabled", True) and a.get("eligible", True)
        ]
    except Exception:
        return []


def _clean_agy_models(stdout: str) -> list[str]:
    """Parse `agy models` output into deduplicated base names (Gemini effort suffixes folded)."""
    clean_models: list[str] = []
    seen: set[str] = set()
    for raw_line in (stdout or "").strip().splitlines():
        line = raw_line.strip()
        if not line or "fetching" in line.lower():
            continue
        parts = line.split()
        if not parts or not any(c in parts[0].lower() for c in ("gemini", "claude", "gpt", "model")):
            continue
        m = base = parts[0]
        for suffix in ("-high", "-medium", "-low"):
            if m.endswith(suffix) and (m.startswith("gemini-") or "flash" in m or "pro" in m):
                base = m[:-len(suffix)]
                break
        if base not in seen:
            seen.add(base)
            clean_models.append(base)
    return clean_models


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
            # agy rejects --effort for Claude models. Offering effort levels
            # here makes Hermes send the flag; the worker dies with a
            # BrokenPipeError and Hermes silently falls back to OpenRouter.
            # (Issue #15)
            return ()
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
            try:
                from .accounts import list_models_for_home
            except ImportError:
                from accounts import list_models_for_home

            # Model access is per Google account. The host default account (real HOME)
            # serves any model no rotation account has (pick_account skips accounts
            # without the model), so offer the union of every usable account's models.
            homes = [None] + [a["home_dir"] for a in _rotation_accounts() if a.get("home_dir")]
            raw: list[str] = []
            for home in homes:
                raw.extend(list_models_for_home(home, timeout=timeout))
            clean_models = _clean_agy_models("\n".join(raw))
            if clean_models:
                return clean_models
        except Exception as exc:
            logger.debug("Antigravity fetch_models failed: %s", exc)

        return list(_FALLBACK_MODELS)

    def get_model_context_length(self, model: str) -> int | None:
        """Declared context window for Antigravity CLI.
        
        Defaults to 200,000 tokens (allowing conversations to comfortably pass
        140k-160k tokens before auto-compression). Configurable via the
        ANTIGRAVITY_CONTEXT_LENGTH environment variable.
        """
        env_val = os.environ.get("ANTIGRAVITY_CONTEXT_LENGTH")
        if env_val:
            try:
                val = int(env_val.strip())
                if val > 0:
                    return val
            except ValueError:
                pass
        return 200_000

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
    recovery pipeline triggers auto-compression and retry instead of failing.
    """
    err_str = f"{error} {message}".lower()
    if any(
        pattern in err_str
        for pattern in (
            "subscriber fell behind updates",
            "stalled for 5s",
            "empty result (status='success')",
            "empty result (status=\"success\")",
            "context canceled",
            "max_trajectory_tokens",
            "max trajectory tokens",
        )
    ):
        return {
            "reason": "context_overflow",
            "retryable": True,
            "should_compress": True,
        }
    return None


def _shorten_home(path_str: str) -> str:
    """Return ~ relative path if under user home directory."""
    try:
        p = Path(path_str).resolve()
        h = Path.home().resolve()
        if p == h:
            return "~"
        if p.is_relative_to(h):
            return f"~/{p.relative_to(h)}"
    except Exception:
        pass
    return str(path_str)


def antigravity_auth_handler(action: str, args: Any) -> bool:
    """Provider-owned auth handler for `hermes auth add|status|logout|refresh`.

    Called by Hermes Agent when managing credentials for Antigravity.
    Returns True when the action was handled by this plugin, False to fall through.
    """
    if action == "add":
        if not sys.stdin.isatty():
            raise SystemExit(
                "Error: 'hermes auth add antigravity-subscription-directsdk' requires an interactive terminal (TTY) "
                "so you can complete the browser-based Google authentication."
            )

        user_label = (getattr(args, "label", None) or "").strip()
        accounts_dir = get_accounts_dir()
        accounts_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            accounts_dir.chmod(0o700)

        is_staging = False
        if user_label:
            folder_name = sanitize_folder_name(user_label)
            home_dir = accounts_dir / folder_name
        else:
            folder_name = f".staging_{os.getpid()}_{int(time.time() * 1000)}"
            home_dir = accounts_dir / folder_name
            is_staging = True

        home_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            home_dir.chmod(0o700)

        try:
            cmd = resolve_agy_command()
        except Exception as exc:
            if is_staging:
                shutil.rmtree(home_dir, ignore_errors=True)
            raise SystemExit(f"Error resolving agy CLI: {exc}")

        env = os.environ.copy()
        env["HOME"] = str(home_dir)
        env.pop("ANTIGRAVITY_CONFIG_DIR", None)

        print(f"Opening agy in {home_dir} for Antigravity sign-in...")
        print("Please complete the sign-in prompt in your browser if requested.")
        print(
            "Note: On macOS, if a dialog appears saying 'a keychain cannot be found to store \"antigravity\"', "
            "you can safely click Cancel. Authentication tokens are stored directly in account files."
        )

        # Do not raise on returncode != 0 (macOS keychain dialog cancel causes non-zero exit).
        # Output stays inherited so the browser prompt remains visible; the timeout
        # only stops a sign-in that never finishes from hanging the CLI forever.
        try:
            login_proc = subprocess.run([cmd], env=env, check=False, timeout=_LOGIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if is_staging:
                shutil.rmtree(home_dir, ignore_errors=True)
            raise SystemExit(
                "Error: Antigravity sign-in did not finish within "
                f"{int(_LOGIN_TIMEOUT_SECONDS // 60)} minutes; nothing was registered."
            )
        rc = getattr(login_proc, "returncode", 0)
        if isinstance(rc, int) and rc != 0:
            print(f"Note: agy exited with code {rc}.")

        token_path = find_account_token_path(home_dir)
        if not token_path:
            if is_staging:
                shutil.rmtree(home_dir, ignore_errors=True)
            raise SystemExit("Error: Authentication failed. No OAuth token file was found after running agy.")

        extracted_email = extract_email_from_token_file(token_path)
        if user_label:
            final_label = user_label
            final_home = home_dir
        else:
            if extracted_email:
                final_label = extracted_email
            else:
                if sys.stdin.isatty():
                    try:
                        final_label = input("Account label: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        final_label = ""
                else:
                    final_label = ""
                if not final_label:
                    existing_labels = {acc.get("label") for acc in list_accounts()}
                    idx = len(existing_labels) + 1
                    while f"account-{idx}" in existing_labels:
                        idx += 1
                    final_label = f"account-{idx}"

            folder_name = sanitize_folder_name(final_label)
            final_home = accounts_dir / folder_name
            if home_dir != final_home:
                if final_home.exists():
                    shutil.rmtree(final_home, ignore_errors=True)
                shutil.move(str(home_dir), str(final_home))
                with contextlib.suppress(OSError):
                    final_home.chmod(0o700)

        # Check eligibility
        is_eligible, usage_data = check_account_eligibility(final_home)

        add_account(
            label=final_label,
            home_dir=str(final_home),
            enabled=True,
            eligible=is_eligible,
            email=extracted_email,
        )

        print(f"Successfully saved account '{final_label}' to registry.")
        if is_eligible:
            # Eligible accounts stay quiet: no warning, just the current quota.
            if usage_data:
                g_5h = usage_data.get("gemini", {}).get("5h", {}).get("remaining_fraction")
                g_wk = usage_data.get("gemini", {}).get("weekly", {}).get("remaining_fraction")
                c_5h = usage_data.get("claude_gpt", {}).get("5h", {}).get("remaining_fraction")
                c_wk = usage_data.get("claude_gpt", {}).get("weekly", {}).get("remaining_fraction")
                if g_5h is not None and g_wk is not None:
                    print(f"  Gemini Quota (5h / Wk): {int(g_5h * 100)}% / {int(g_wk * 100)}%")
                if c_5h is not None and c_wk is not None:
                    print(f"  Claude/GPT Quota (5h / Wk): {int(c_5h * 100)}% / {int(c_wk * 100)}%")
        else:
            print("WARNING: this account is NOT eligible for the Antigravity subscription.")
            print("WARNING: it stays registered but quota rotation will skip it.")

        return True

    if action == "status":
        accounts = list_accounts()
        active = get_active()
        rotation_mode = get_rotation_mode()

        mode_source = "env ANTIGRAVITY_ROTATION" if os.environ.get("ANTIGRAVITY_ROTATION", "").strip() else "registry"
        print(f"Rotation mode: {rotation_mode} (source: {mode_source})")
        if rotation_mode == "fixed":
            print(f"Pinned account: {active or '(none; using host default)'}")

        headers = ["Label", "Active", "Enabled", "Eligible", "Cooldown", "Gemini (5h / Wk)", "Claude/GPT (5h / Wk)"]
        row_format = "{:<22} {:<8} {:<9} {:<10} {:<12} {:<20} {:<20}"
        divider = "-" * 105
        print(row_format.format(*headers))
        print(divider)

        now = time.time()

        # Host default account
        host_home = Path.home()
        host_usage = fetch_usage_for_home(str(host_home), cached=False)
        is_host_active = (rotation_mode == "off") or (not accounts) or (active == "(host default)")

        host_act_str = "*" if is_host_active else ""
        host_en_str = "yes"
        host_elig_str = "yes" if host_usage is not None else "unknown"
        host_cd_str = "ready"

        if host_usage:
            g_5h = host_usage.get("gemini", {}).get("5h", {}).get("remaining_fraction")
            g_wk = host_usage.get("gemini", {}).get("weekly", {}).get("remaining_fraction")
            c_5h = host_usage.get("claude_gpt", {}).get("5h", {}).get("remaining_fraction")
            c_wk = host_usage.get("claude_gpt", {}).get("weekly", {}).get("remaining_fraction")
            g_str = f"{int(g_5h * 100)}% / {int(g_wk * 100)}%" if g_5h is not None and g_wk is not None else "n/a"
            c_str = f"{int(c_5h * 100)}% / {int(c_wk * 100)}%" if c_5h is not None and c_wk is not None else "n/a"
        else:
            g_str = "unknown"
            c_str = "unknown"

        print(row_format.format("(host default)", host_act_str, host_en_str, host_elig_str, host_cd_str, g_str, c_str))
        print(f"  Home: {_shorten_home(str(host_home))}")

        for acc in accounts:
            lbl = acc.get("label", "")
            is_act = "*" if (not is_host_active and lbl == active) else ""
            en = "yes" if acc.get("enabled", True) else "no"
            elig = "yes" if acc.get("eligible", True) else "no"
            cd_until = acc.get("cooldown_until", 0.0)
            if cd_until > now:
                cd_str = f"{int(cd_until - now)}s"
            else:
                cd_str = "ready"

            usage = fetch_usage_for_home(acc.get("home_dir", ""), cached=False)
            if usage:
                g_5h = usage.get("gemini", {}).get("5h", {}).get("remaining_fraction")
                g_wk = usage.get("gemini", {}).get("weekly", {}).get("remaining_fraction")
                c_5h = usage.get("claude_gpt", {}).get("5h", {}).get("remaining_fraction")
                c_wk = usage.get("claude_gpt", {}).get("weekly", {}).get("remaining_fraction")
                gemini_str = f"{int(g_5h * 100)}% / {int(g_wk * 100)}%" if g_5h is not None and g_wk is not None else "n/a"
                claude_str = f"{int(c_5h * 100)}% / {int(c_wk * 100)}%" if c_5h is not None and c_wk is not None else "n/a"
            else:
                gemini_str = "unknown"
                claude_str = "unknown"

            print(row_format.format(lbl, is_act, en, elig, cd_str, gemini_str, claude_str))
            print(f"  Home: {_shorten_home(acc.get('home_dir', ''))}")

        if not accounts:
            print("\n(No additional Antigravity accounts registered.)")
            print("Run `hermes auth add antigravity-subscription-directsdk` to add an account.")

        # Warn only about accounts that cannot be used; eligible accounts stay quiet.
        ineligible = [acc.get("label", "") for acc in accounts if not acc.get("eligible", True)]
        if ineligible:
            print(f"\nWARNING: not eligible for Antigravity (skipped by rotation): {', '.join(ineligible)}")

        return True

    if action in ("logout", "remove"):
        target = (getattr(args, "target", None) or getattr(args, "label", None) or "").strip()
        if target:
            if remove_account(target):
                print(f"Account '{target}' removed from registry. Home directory preserved.")
                return True
            else:
                raise SystemExit(f"Error: Account '{target}' not found in registry.")

        accounts = list_accounts()
        if not accounts:
            print("No Antigravity accounts registered.")
            return True

        # Without a TTY there is no way to ask which account to drop, and
        # removing one (or all) of them silently from a script is a footgun:
        # require an explicit label instead.
        if not sys.stdin.isatty():
            raise SystemExit(
                "Error: refusing to remove registered Antigravity accounts from a "
                "non-interactive session with no account label given. Pass the label "
                "to remove, or run the command from a terminal."
            )

        if len(accounts) == 1:
            lbl = accounts[0]["label"]
            remove_account(lbl)
            print(f"Account '{lbl}' removed from registry. Home directory preserved.")
            return True

        if sys.stdin.isatty():
            print("Registered Antigravity accounts:")
            for i, acc in enumerate(accounts, 1):
                print(f"  {i}. {acc['label']}")
            try:
                choice = input("Enter label to remove (or 'all' to remove all): ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(1)
            if choice.lower() == "all":
                for acc in accounts:
                    remove_account(acc["label"])
                print(f"All {len(accounts)} accounts removed from registry. Home directories preserved.")
                return True
            elif choice:
                if remove_account(choice):
                    print(f"Account '{choice}' removed from registry. Home directory preserved.")
                    return True
                else:
                    raise SystemExit(f"Error: Account '{choice}' not found in registry.")
            else:
                raise SystemExit("Error: No account selected.")

        # Defensive: the guard above already rejects every non-interactive path,
        # and the interactive path above always returns. Keep the refusal here so
        # a future edit cannot silently re-introduce an unattended mass removal.
        raise SystemExit(
            "Error: refusing to remove every registered Antigravity account from a "
            "non-interactive session. Pass the account label to remove, or run the "
            "command from a terminal."
        )

    if action == "refresh":
        target = (getattr(args, "target", None) or getattr(args, "label", None) or "").strip()
        if target:
            accounts = {acc["label"]: acc for acc in list_accounts()}
            if target not in accounts:
                raise SystemExit(f"Error: Account '{target}' not found in registry.")
            set_cooldown(target, until=0.0)
            fetch_usage_for_home(accounts[target].get("home_dir", ""), cached=False)
            print(f"Cleared cooldown and refreshed quota for '{target}'.")
            return True

        accounts = list_accounts()
        if not accounts:
            print("No Antigravity accounts registered.")
            return True
        for acc in accounts:
            lbl = acc["label"]
            set_cooldown(lbl, until=0.0)
            fetch_usage_for_home(acc.get("home_dir", ""), cached=False)
        print(f"Cleared cooldowns and refreshed quota for {len(accounts)} account(s).")
        return True

    if action == "use":
        target = (getattr(args, "target", None) or getattr(args, "label", None) or "").strip()
        if not target:
            raise SystemExit("Error: target/label is required.")
        if set_active(target):
            print(f"Active account set to '{target}'.")
            return True
        else:
            raise SystemExit(f"Error: Account '{target}' not found in registry.")

    return False


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
    classify_api_error=_classify_antigravity_error,
    auth_handler=antigravity_auth_handler,
)

auth_handler = antigravity_auth_handler
register_provider(antigravity_profile)
