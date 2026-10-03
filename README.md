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
- **Error classification and auto-compression**: `classify_api_error` detects `agy` PubSub stalls (`subscriber fell behind updates, stalled for 5s`), `context canceled`, and empty SUCCESS status. Returns `context_overflow` with `should_compress: True`, so Hermes compresses the session and retries instead of failing.

---

## Models

| Model | Suffix | LLM Context | Plugin Declared | Supported Efforts |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | `-low`, `-medium`, `-high` | 1M tokens | 200k tokens | `low`, `medium`, `high` |
| `gemini-3.7-flash` | `-low`, `-medium`, `-high` | 1M tokens | 200k tokens | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `-low`, `-medium`, `-high` | 1M tokens | 200k tokens | `low`, `medium`, `high` |
| `gemini-3.1-pro` | `-low`, `-high` | 2M tokens | 200k tokens | `low`, `high` |
| `claude-sonnet-4-6` | None | 200k tokens | 200k tokens | None (agy rejects `--effort`) |
| `claude-opus-4-6-thinking` | None | 200k tokens | 200k tokens | None (agy rejects `--effort`) |
| `gpt-oss-120b-medium` | None | 128k tokens | 200k tokens | None |

> **LLM Context vs Plugin Declared**: The LLM context column shows the model's native token window. The plugin declares 200,000 tokens to Hermes (configurable via `ANTIGRAVITY_CONTEXT_LENGTH`). This gap exists because `agy` runs an internal Go language server (`jetski/cortex`) that re-serializes the cumulative trajectory on each token via a gRPC channel with a 5-second drain deadline. Prompts exceeding 500 KB cause channel backpressure that trips the deadline and drops the stream. The 200k declared limit triggers Hermes auto-compression at 80% (160k tokens), keeping wire payloads within `agy` throughput limits.

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
| `gpt-oss-120b-medium` | Follows | Tested with full toolset |

Models that ignore the protocol attempt `agy`'s native `RunCommand`/`WriteToFile` steps. The plugin neutralizes these (stream watcher kills the process), but every tool-using turn fails. This is an upstream `agy` limitation: the persona prompt is embedded in the closed binary with no configuration surface to suppress it.

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
| `ANTIGRAVITY_CONTEXT_LENGTH` | `200000` | Override the declared context window (tokens). Hermes triggers auto-compression at 80% of this value. |
| `ANTIGRAVITY_COMMAND` | `agy` | Path to the `agy` binary. Also checks `AGY_CLI_PATH` and `ANTIGRAVITY_CLI_PATH`. |
| `ANTIGRAVITY_ARGS` | (none) | Extra arguments to pass to the `agy` subprocess. |
| `ANTIGRAVITY_CONFIG_DIR` | (none) | Override the config directory, bypassing keyring and token-file detection. |
| `ANTIGRAVITY_ROTATION` | `off` | Multi-account rotation mode: `off` (default), `quota`, or `round_robin`. |
| `ANTIGRAVITY_ACCOUNTS_DIR` | `~/.agy-accounts` | Base directory for multi-account isolated HOME environments. |
| `ANTIGRAVITY_ACCOUNTS_FILE` | `~/.hermes/antigravity-accounts.json` | Path to the multi-account registry file. |

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

### Multi-Account & Quota Rotation

When you have multiple Google accounts with Antigravity / Gemini subscriptions, you can register them and rotate between them automatically.

```bash
# Add an account (label is optional; if omitted, defaults to the account email from id_token)
hermes auth add antigravity-subscription-directsdk

# Or specify an explicit label
hermes auth add antigravity-subscription-directsdk --label work

# Check status and remaining quotas across accounts (includes host default account)
hermes auth status antigravity-subscription-directsdk

# Refresh quota and clear cooldowns
hermes auth refresh antigravity-subscription-directsdk

# Remove an account from registry (preserves credentials and directory)
hermes auth logout antigravity-subscription-directsdk work
```

#### Account Setup & Authentication

- **Optional Label & Safe Directory Names**: `--label` is optional. When omitted, the plugin extracts the email address directly from the JWT `id_token` payload in the newly saved token file. The home directory name is sanitized for cross-platform compatibility (`[A-Za-z0-9._-]`), while the saved label retains the full email address.
- **Token File-based Authentication**: Success is determined by the presence of the OAuth token file (`antigravity-oauth-token` or `jetski-standalone-oauth-token`), rather than the process exit code.
- **macOS Keychain Dialog**: On macOS, a system dialog may prompt: *"a keychain cannot be found to store 'antigravity'"*. You can safely click **Cancel**. Account credentials are stored securely in isolated token files and do not affect the host keychain.
- **Automatic Eligibility Check**: Upon sign-in, the plugin validates subscription eligibility via `agy -p /usage --output-format json`. Ineligible accounts are saved with `eligible: false` and are automatically bypassed during quota rotation.

#### Rotation Modes

Configured via `ANTIGRAVITY_ROTATION` environment variable in your `~/.hermes/config.yaml` or shell environment:

- `off` (default): Always use the host default account or the single active account.
- `quota`: Automatically pick the healthiest account on each request. Ineligible accounts, accounts on cooldown, and accounts with 0% remaining quota in either window are hard-gated (score = 0.0). The score is `f_5h * (f_weekly ** 2)` scaled by how soon each window resets (a window about to reset is worth spending) and divided by the number of in-flight turns on that account, so parallel work spreads out. An account whose quota could not be read is scored as half-full and damped, so it never outranks an account with a known healthy quota.
- `round_robin`: Cycle through available, eligible accounts sequentially.
- `fixed`: Always use the pinned account (`hermes antigravity use <label>`); falls back to the host default if that account is unusable.

If an account encounters a quota error (HTTP 429 / resource exhausted), it is placed on a cooldown until the window resets and Hermes automatically fails over to the next best available account — including while rotation is `off`, so a depleted account swaps accounts instead of degrading to another model. Transient rate limiting alone does not trigger a swap.

Account credentials are stored under `~/.agy-accounts/<label>/` and the registry file is stored at `~/.hermes/antigravity-accounts.json` (configurable via `ANTIGRAVITY_ACCOUNTS_DIR` and `ANTIGRAVITY_ACCOUNTS_FILE`).

---

## Tests

Run the test suite:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
```

---

## License

MIT
