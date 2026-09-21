# Antigravity Subscription DirectSDK Provider for Hermes Agent

[![Hermes Agent Plugin](https://img.shields.io/badge/Hermes%20Agent-Plugin-purple.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tested with Hermes](https://img.shields.io/badge/Hermes-v0.21%2B-green.svg)](https://github.com/nousresearch/hermes-agent)

An enterprise-grade model provider plugin for **[Hermes Agent](https://github.com/nousresearch/hermes-agent)** that connects directly to your existing Google Antigravity / Gemini subscription via the official `agy` CLI binary.

Run top-tier reasoning and frontier models (**Gemini 3.8 Flash**, **Gemini 3.1 Pro**, **Claude Sonnet 4.6**, **Claude Opus 4.6 Thinking**) inside Hermes without paying for extra API credits or managing third-party API keys.

---

## Key Features

- ⚡ **Progressive Real-Time Streaming**: Delivers instant token-by-token streaming responses to Hermes using the low-latency `stream-json` engine.
- 🛡️ **Hermes Tool Monopoly & Security**: Antigravity's 56 native execution tools (`run_command`, `write_to_file`, `view_file`, etc.) are completely neutralized. Hermes maintains 100% control over tool parsing, safety validation, user confirmation, and host execution.
- 🧠 **Adaptive Thinking Effort Picker**: Seamlessly integrates into Hermes's interactive `/model` picker. Automatically filters supported reasoning efforts (`low`, `medium`, `high`) per model and dynamically maps them to backend quota slugs.
- 🔀 **Multi-Agent & Subagent Concurrency**: Fully thread-safe design with isolated POSIX process sessions (`start_new_session=True`). Spawns parallel, non-blocking processes for concurrent subagents with zero memory leaks.
- 🧼 **Context Isolation**: Executes in a clean, isolated temporary workspace (`/tmp`). Automatically prevents host `GEMINI.md`, `AGENTS.md`, or repository-level rules from contaminating Hermes's system prompt.
- 📉 **Zero Resident Memory Overhead**: Uses native compiled Go process invocations (~119ms startup overhead, <4% total turn latency). Completely terminates when idle with 0MB background memory footprint.

---

## Supported Models

| Model Slug | Provider Suffix | Context Window | Supported Reasoning Efforts | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`gemini-3.8-flash`** | `-low`, `-medium`, `-high` | 1,048,576 tokens | `low`, `medium`, `high` | Flagship multimodal reasoning model, ultra-fast latency. |
| **`gemini-3.7-flash`** | `-low`, `-medium`, `-high` | 1,048,576 tokens | `low`, `medium`, `high` | Prior generation high-efficiency reasoning model. |
| **`gemini-3.6-flash`** | `-low`, `-medium`, `-high` | 1,048,576 tokens | `low`, `medium`, `high` | Lightweight, instant-response model for routine tasks. |
| **`gemini-3.1-pro`** | `-low`, `-high` | 2,097,152 tokens | `low`, `high` | Deep reasoning and large-codebase architectural analysis. |
| **`claude-sonnet-4-6`** | None | 200,000 tokens | Standard | High-precision coding and agentic planning. |
| **`claude-opus-4-6-thinking`** | None | 200,000 tokens | Extended Thinking | Complex multi-step reasoning and mathematical analysis. |
| **`gpt-oss-120b-medium`** | None | 128,000 tokens | Standard | Open-weight foundation model option. |

---

## Security & Sandboxing Architecture

Antigravity CLI by default registers local host-execution tools. To ensure Hermes Agent remains the sole executor and arbiter of tool calls, this plugin enforces a **4-layer containment architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                        Hermes Agent                         │
│  - System Prompt & Personality                              │
│  - Tool Registry & Safety Approvals                         │
│  - Tool Execution & State Tracking                          │
└──────────────┬───────────────────────────────▲──────────────┘
               │ Prompt + Tool Schemas         │ OpenAI-compatible
               │                               │ <tool_call> tokens
┌──────────────▼───────────────────────────────┴──────────────┐
│       Antigravity Subscription DirectSDK Plugin             │
│                                                             │
│  Layer 1: Precedence Prompt Preamble                        │
│           (Disables model-initiated local tool calling)     │
│  Layer 2: Stripped `--dangerously-skip-permissions`         │
│           (Enforces agy strict request-review headless mode)│
│  Layer 3: Real-Time Stream Watchdog                         │
│           (Kills process immediately on native 'tool' step) │
│  Layer 4: Neutral CWD Isolation                             │
│           (Runs in /tmp; blocks GEMINI.md / AGENTS.md leak) │
└──────────────────────────────┬──────────────────────────────┘
                               │ Headless stream-json (pipes)
┌──────────────────────────────▼──────────────────────────────┐
│           Official `agy` CLI Binary (Go Native)             │
│           Google Antigravity Cloud Backend                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Official Antigravity CLI (`agy`)**:
   Ensure `agy` is installed and authenticated on your machine:
   ```bash
   # Verify agy is installed and logged in
   agy --version
   ```
2. **Hermes Agent**:
   Requires Hermes Agent version `0.21.0` or higher:
   ```bash
   hermes --version
   ```

---

## Installation

### Option 1: Via Hermes Plugin Manager (Recommended)

```bash
# Install directly from GitHub
hermes plugins install https://github.com/soyelmismo/hermes-antigravity-subscription

# Enable the plugin
hermes plugins enable antigravity-subscription-directsdk
```

### Option 2: Manual Installation

Clone or copy this repository into your local Hermes plugins directory:

```bash
git clone https://github.com/soyelmismo/hermes-antigravity-subscription.git \
  ~/.hermes/plugins/antigravity-subscription-directsdk

hermes plugins enable antigravity-subscription-directsdk
```

---

## Usage

### 1. Interactive Switch via `/model`

Launch Hermes and run `/model`:
```text
/model
```
1. Select **Antigravity Subscription DirectSDK** as the provider.
2. Choose your desired model (e.g. `gemini-3.8-flash`).
3. Select your desired reasoning effort (`low`, `medium`, `high`).

### 2. Configuration via CLI Flags or Config

You can also specify the model on startup:

```bash
hermes --provider antigravity-subscription-directsdk --model gemini-3.8-flash
```

Or configure it as default in `~/.hermes/config.yaml`:

```yaml
agent:
  provider: antigravity-subscription-directsdk
  model: gemini-3.8-flash
  reasoning_effort: high
```

---

## Verification & Unit Testing

The repository includes a comprehensive test suite covering provider registration, effort resolution, streaming token parsing, XML `<tool_call>` extraction, security guards, and process watchdog:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. python3 tests/test_antigravity_plugin.py
```

Expected output:
```text
...............
----------------------------------------------------------------------
Ran 15 tests in 1.14s

OK
```

---

## Plugin Catalog Submission

To submit this plugin to the official Hermes Agent Plugin Catalog (`nousresearch/hermes-agent`), use the validated catalog entry in [`catalog-entry.yaml`](catalog-entry.yaml):

```yaml
name: antigravity-subscription-directsdk
repo: https://github.com/soyelmismo/hermes-antigravity-subscription
sha: <COMMIT_SHA>
description: "Run Hermes on an Antigravity / Gemini subscription via the official agy CLI (DirectSDK path). Streams tokens progressively, maps thinking effort, and ensures safe sandboxed execution."
maintainer: soyelmismo
tier: community
category: models
requires_hermes: ">=0.21.0"
docs_url: https://github.com/soyelmismo/hermes-antigravity-subscription#readme
version: "1.0.0"
platforms:
  - linux
  - macos
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env: []
```

Validate the entry against Hermes's catalog validator:
```bash
python3 scripts/validate_plugin_catalog.py catalog-entry.yaml
```

---

## License

MIT License. Copyright (c) 2026 soyelmismo. See [LICENSE](LICENSE) for details.
