# Antigravity Subscription DirectSDK Provider for Hermes Agent

[![Hermes Agent Plugin](https://img.shields.io/badge/Hermes%20Agent-Plugin-purple.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tested with Hermes](https://img.shields.io/badge/Hermes-v0.21%2B-green.svg)](https://github.com/nousresearch/hermes-agent)

Hermes Agent model provider that routes inference through an active Google Antigravity account using the local `agy` binary.

This plugin lets Hermes use Gemini and Claude models through your existing Antigravity quota.

---

## How It Works

- **Token streaming**: Reads stdout chunks from `agy --output-format stream-json` and yields standard completion deltas.
- **Tool call routing**: The model emits `<tool_call>` tags in text. The plugin parses these into OpenAI function call deltas, so Hermes executes tools on the host instead of `agy`.
- **Tool execution guard**: `agy` registers local system tools by default. This plugin runs `agy` in headless mode without permissions skip flags. If `agy` attempts to execute an internal tool step, the stream closes and terminates the child process.
- **Thinking effort mapping**: Maps Hermes reasoning effort settings (`low`, `medium`, `high`) to backend model variants (`gemini-3.8-flash-low`, `gemini-3.8-flash-high`).
- **Subagent concurrency**: Each completion turn runs in its own process group (`start_new_session=True`). Multiple Hermes subagents can request completions concurrently without shared state.
- **Filesystem isolation**: Subprocesses run in an isolated, private temporary working directory per client and slash commands are disabled. Local `GEMINI.md` and `AGENTS.md` project files are not read.
- **Tool output pruning**: Old tool results (diffs, directory trees, test output) dominate wire payload in long sessions. The prompt builder keeps the last 8 tool results intact and caps older tool outputs at 300 characters. A 20k character safety limit with head/tail preservation guards individual tool calls. In a 336-message session, this cut wire payload from 600 KB to 180 KB (70% reduction).
- **Error classification and recovery**: `classify_api_error` detects `agy` PubSub stalls (`subscriber fell behind updates, stalled for 5s`), trajectory limits, and empty SUCCESS status (returning `context_overflow` with `should_compress: True`). It classifies credit balance exhaustion and billing caps as `billing` for immediate failover, and maps account verification and Terms of Service notices to `auth_permanent` to surface resolution links without retry loops.

---

## Models

| Model | Suffix | LLM Context | Plugin Declared | Supported Efforts |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.7-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.1-pro` | `-low`, `-high` | 2M tokens | 239k tokens | `low`, `high` |
| `gpt-oss-120b` | `-medium` only | 128k tokens | 128k tokens | `medium` |
| `claude-sonnet-4-6` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |
| `claude-opus-4-6-thinking` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |

**Dynamic discovery per account.** Available models and supported reasoning efforts are discovered dynamically from `agy models`. The picker reflects the exact models available on the user's account and subscription tier (including preview models or custom models configured in `agy` settings). A model that agy lists by bare name takes no `--effort`. A requested effort that the model lacks maps to the nearest supported one, the stronger on a tie (`medium` on `gemini-3.1-pro` becomes `high`). The result is cached for one hour; `models.py` provides an emergency fallback when `agy models` fails. Hermes' own `xhigh` and `max` map to `high`.

> **Claude Model Aliases**: Requests for `claude-sonnet-5-5` or `claude-opus-5-5` resolve dynamically if present in the user's `agy models` catalog; if the account lacks them, they gracefully fall back to `claude-sonnet-4-6` and `claude-opus-4-6-thinking`.

> **Trajectory Persistence & Context Length**: Multi-turn sessions persist across worker restarts via `agy --conversation <id>`, sending only incremental deltas on each turn. This bypasses `agy`'s single-turn 100,000-token prompt trimmer. The plugin declares 239,000 tokens to Hermes (configurable via `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` or `ANTIGRAVITY_CONTEXT_LENGTH`), ensuring Hermes auto-compaction triggers safely below `agy`'s internal checkpoint threshold (239,616 tokens / 256k - 16k output tokens) and preserves sovereign context control. Conversation databases are strictly isolated inside the worker's private temp directory and automatically purged on session resets (`/new`).

### Model Compatibility

`agy` embeds its own system prompt ("you are Antigravity, you have `run_command`, `view_file`, ...") before the plugin's preamble. Some models obey `agy`'s prompt over the plugin's `<tool_call>` protocol and invoke native tools, which headless mode soft-denies. The plugin detects this and terminates the stream, but the turn is lost.

| Model | `<tool_call>` Protocol | Notes |
| :--- | :--- | :--- |
| `gemini-3.8-flash` | Follows | Tested with full toolset (~40 tools) |
| `gemini-3.7-flash` | Follows | Tested with full toolset |
| `gemini-3.6-flash` | Ignores | Goes native even with a single tool schema |
| `gemini-3.1-pro` | Follows | Tested with full toolset |
| `claude-opus-4-6-thinking` | Follows | Tested with full toolset |
| `claude-sonnet-4-6` | Partial | Works with moderate toolsets (tested up to 10 tools); refuses or flips to native with large toolsets (~40 tools) |
| `gpt-oss-120b` | Follows | Tested with full toolset |

Models that ignore the protocol attempt `agy`'s native `RunCommand`/`WriteToFile` steps. The plugin neutralizes these (stream watcher kills the process), but every tool-using turn fails. This is an upstream `agy` limitation: the persona prompt is embedded in the closed binary with no configuration surface to suppress it.

---

## Environment passed to `agy`

Hermes often runs with gateway and dashboard secrets in its environment (`SLACK_BOT_TOKEN`, `TELEGRAM_BOT_TOKEN`, `*_API_KEY`, ...). `agy` is a closed-source binary and does not need any of them, so by default the child environment is the parent environment **minus credential-looking variables**: names containing `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `ACCESS_KEY`, `PRIVATE_KEY`, `CREDENTIAL(S)`, `AUTH`, `WEBHOOK`, `DSN` or `COOKIE`, and database/broker URLs (`DATABASE_URL`, `REDIS_URL`, `MONGO_URI`, ...). `SSH_AUTH_SOCK`, proxy and certificate variables are kept.

This default is a **name heuristic**: it cannot know what an arbitrarily named variable holds. For a guarantee of exactly what `agy` sees, use strict mode.

| Variable | Effect |
| :--- | :--- |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | Comma-separated names to keep even if they look like credentials. |
| `ANTIGRAVITY_ENV_STRICT` | `1`, `true`, `yes` or `on`: pass only the baseline below plus `ANTIGRAVITY_ENV_ALLOWLIST` and `ANTIGRAVITY_ENV_PASSTHROUGH`. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | Comma-separated names to add in strict mode. |

Strict baseline: `PATH`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`/`TEMP`/`TMP`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (and their lowercase spellings on POSIX), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS` and `XDG_RUNTIME_DIR` (Linux keyring sign-in), `SYSTEMROOT`, `WINDIR`, `COMSPEC`, `PATHEXT` (Windows), `AGY_CLI_DISABLE_AUTO_UPDATE` and `AGY_CLI_MODEL_API_MAX_RETRIES`. Names are matched case-insensitively on Windows.

---

## Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                        Hermes Agent                         │
│  - Prompts and system instructions                          │
│  - Tool definitions and approvals                           │
│  - Host execution                                           │
└──────────────┬───────────────────────────────▲──────────────┘
               │ Prompt + tool schemas         │ Text with <tool_call> tags
               │                               │ parsed into completion chunks
┌──────────────▼───────────────────────────────┴──────────────┐
│       Antigravity Subscription DirectSDK Plugin             │
│                                                             │
│  1. Preamble tells the model to emit tool tags in text.     │
│  2. Omits --dangerously-skip-permissions to deny agy tools. │
│  3. Stream watcher kills agy if it emits a tool step.       │
│  4. Runs in private temp dir to ignore rules files.         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Stdin / stdout (NDJSON pipes)
┌──────────────────────────────▼──────────────────────────────┐
│               Local `agy` CLI binary (Go)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Requirements

1. **Antigravity CLI**:
   `agy` must be installed and authenticated on the system.
   ```bash
   agy --version
   ```
2. **Hermes Agent**:
   Version `0.21.0` or higher.
   ```bash
   hermes --version
   ```

> **Optional: Linux keyring detection.**
> On Linux with a D-Bus session, `agy` keeps its session in the freedesktop
> Secret Service instead of a token file (credentials `service`/`gemini` +
> `username`/`antigravity`, the go-keyring convention). Detection uses
> `secret-tool`, falling back to `python3` with `secretstorage` when the CLI is
> absent or cannot answer (missing, spawn failure, timeout or non-zero exit).
> `secret-tool` prints the credential on its own stdout, so the plugin closes
> that stream and reads only the attribute lines on stderr; the scripted
> fallback reads attributes and labels and never asks for the secret. Either way
> the plugin never receives the stored secret. Installing `secret-tool` is not
> strictly neutral: the CLI path answers on the exact attribute pair, while the
> scripted path additionally matches a credential whose go-keyring label embeds
> "antigravity". Headless systems without a D-Bus session keep using the token-file
> path, so nothing extra is required there.
> ```bash
> command -v secret-tool   # optional: faster than the python3 fallback
> ```
> Install it with `libsecret-tools` (Debian/Ubuntu) or `libsecret` (Fedora/Arch).

> **SSH session variables.**
> `agy` switches to file-based token storage when it detects `SSH_CONNECTION`,
> `SSH_CLIENT`, or `SSH_TTY`. When no token file exists (credential lives in the
> OS keyring), the plugin strips these variables from the child environment so
> `agy` uses keyring authentication instead of failing. `SSH_AUTH_SOCK` is
> preserved. When a token file exists, the environment is unchanged.

> **macOS keychain.** On macOS `agy` keeps its session in the login keychain
> (a generic password with service `gemini` and account `antigravity`).
> Detection runs `/usr/bin/security find-generic-password` without `-g` or
> `-w`, which prints only the item's attributes, so the plugin never receives
> the secret and no keychain prompt appears. The keychain is found through
> `$HOME/Library/Keychains`, so the plugin links that directory into the
> isolated HOME it gives `agy`; otherwise the child `agy` would ask you to log
> in again. Nothing extra is required. Setting `ANTIGRAVITY_CONFIG_DIR` turns
> both off and keeps the token-file path.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` | `239000` | Override the declared context window (tokens). Also checks `ANTIGRAVITY_CONTEXT_LENGTH`. Hermes triggers auto-compression based on this value (e.g. at 75% threshold, ~179k tokens). |
| `ANTIGRAVITY_COMMAND` | `agy` | Path to the `agy` binary. Also checks `AGY_CLI_PATH` and `ANTIGRAVITY_CLI_PATH`. |
| `ANTIGRAVITY_ARGS` | (none) | Extra arguments to pass to the `agy` subprocess. |
| `ANTIGRAVITY_CONFIG_DIR` | (none) | Override the config directory, bypassing keyring and token-file detection. |
| `ANTIGRAVITY_DEBUG_PROMPT_DIR` | (none) | Directory to dump outgoing prompts sent to `agy` for offline diagnosis. |
| `ANTIGRAVITY_WORKER_IDLE_SECONDS` | `900` | Terminate the persistent `agy` worker after this many seconds without a finished turn. The next turn starts a fresh worker with the full prompt. `0` or a negative value disables the bound. |
| `ANTIGRAVITY_PROXY` | (none) | Proxy URL (e.g. `http://127.0.0.1:8080` or `socks5h://127.0.0.1:1080`) agy's subprocess should use. Sets `HTTP(S)_PROXY` and `ALL_PROXY` (both casings) and adds loopback to `NO_PROXY`. Unset means agy inherits the parent process's proxy environment unchanged. |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | (none) | Comma-separated variable names to preserve in the child environment even if they match secret heuristics. |
| `ANTIGRAVITY_ENV_STRICT` | `false` | When enabled (`1`, `true`), only passes variables from `ANTIGRAVITY_ENV_ALLOWLIST` plus a minimal runtime baseline. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | (none) | Additional variables to pass through when `ANTIGRAVITY_ENV_STRICT` is active. |

---

## Installation

Install through the Hermes plugin manager:

```bash
hermes plugins install https://github.com/soyelmismo/hermes-antigravity-subscription
hermes plugins enable antigravity-subscription-directsdk
```

Or clone into the local plugins folder:

```bash
git clone https://github.com/soyelmismo/hermes-antigravity-subscription.git \
  ~/.hermes/plugins/antigravity-subscription-directsdk

hermes plugins enable antigravity-subscription-directsdk
```

> **Directory naming tip**: Hermes Agent discovers plugins by scanning directories in `~/.hermes/plugins/` alphabetically. If you keep backup copies (such as `antigravity-subscription-directsdk.backup`), place them outside `~/.hermes/plugins/` so a later-sorting directory does not override the active plugin.

---

## Usage

### Interactive model picker

Open the model picker inside Hermes:

```text
/model
```

Select **Antigravity Subscription DirectSDK**, then pick a model and effort level.

### Command line flag

```bash
hermes --provider antigravity-subscription-directsdk --model gemini-3.8-flash
```

### Configuration file

Add to `~/.hermes/config.yaml`:

```yaml
agent:
  provider: antigravity-subscription-directsdk
  model: gemini-3.8-flash
  reasoning_effort: high
```

### Subscription Quota & Usage Tracking

The provider profile implements `fetch_account_usage()`, exposing subscription quota limits to Hermes `/usage`, the TUI status bar, and the desktop app via `AccountUsageSnapshot`.

Note: Custom plugin tools and slash commands are not available for `kind: model-provider` plugins because Hermes core skips calling `register(ctx)` for model providers (`hermes_cli/plugins_discovery.py:286`, `hermes_cli/plugin_validate.py:253-258`). All quota visibility is provided through the native `fetch_account_usage()` interface.

Quota queries execute `agy -p "/usage" --output-format json` under an isolated HOME environment without consuming any model tokens or inference turns. Results are cached thread-safely for 60 seconds. Refreshes use single-flight execution so concurrent callers share a single subprocess. An unforced query failure serves the stale cache.

---

## Tests

Run the test suite:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
```

---

## License

MIT
