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
- **Thinking effort mapping**: Maps Hermes reasoning effort settings (`low`, `medium`, `high`) directly to backend model variants (`gemini-3.8-flash-low`, `gemini-3.8-flash-high`).
- **Subagent concurrency**: Each completion turn runs in its own process group (`start_new_session=True`). Multiple Hermes subagents can request completions concurrently without shared state.
- **Filesystem isolation**: Subprocesses run with working directory set to `/tmp` and slash commands disabled. Local `GEMINI.md` and `AGENTS.md` project files are not read.

---

## Models

| Model | Suffix | Context | Supported Efforts |
| :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | `-low`, `-medium`, `-high` | 1M tokens | `low`, `medium`, `high` |
| `gemini-3.7-flash` | `-low`, `-medium`, `-high` | 1M tokens | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `-low`, `-medium`, `-high` | 1M tokens | `low`, `medium`, `high` |
| `gemini-3.1-pro` | `-low`, `-high` | 2M tokens | `low`, `high` |
| `claude-sonnet-4-6` | None | 200k tokens | Default |
| `claude-opus-4-6-thinking` | None | 200k tokens | Extended thinking |
| `gpt-oss-120b-medium` | None | 128k tokens | Default |

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
│  4. Runs in /tmp to ignore workspace rules files.           │
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

---

## Installation

Install directly through the Hermes plugin manager:

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

---

## Tests

Run the test suite:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. python3 tests/test_antigravity_plugin.py
```

---

## License

MIT
