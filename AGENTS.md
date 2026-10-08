# AGENTS.md

Operating guidelines for AI coding assistants working in `soyelmismo/hermes-antigravity-subscription`.

---

## 1. Project Architecture

This repository implements the `antigravity-subscription-directsdk` provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). The plugin runs Hermes on Google Antigravity subscriptions by driving the `agy` CLI as a managed subprocess in `--input-format stream-json --output-format stream-json` mode.

### Module Responsibilities
- [`__init__.py`](__init__.py): Registers `AntigravitySubscriptionDirectSDKProfile`. Declares model capabilities, context bounds (`get_model_context_length` defaulting to 239,000 tokens to keep Hermes compaction safely ahead of `agy`'s 239,616 checkpoint threshold while maximizing usable context), and error classification (`classify_api_error`).
- [`client.py`](client.py): Manages persistent worker lifecycle, per-turn delta reuse, thread locks (`_lock`, `_worker_lock`), isolated HOME configuration, and cleanup retry budgets.
- [`prompt.py`](prompt.py): Assembles prompts from Hermes messages, prunes historical tool outputs to avoid Go channel backpressure, formats delta prompts, and translates `<tool_call>` blocks.
- [`process.py`](process.py): Detects the `agy` binary, reads OS keyring tokens (Linux, macOS, Windows), creates isolated HOME symlinks, and terminates process trees.
- [`stream.py`](stream.py): Parses streaming JSON events from `agy`, emits token chunks, detects native tool abort signals, and tracks per-turn usage deltas.
- [`plugin.yaml`](plugin.yaml): Plugin manifest for Hermes Agent and the catalog registry.

---

## 2. Development and Testing

- **Test Suite**: Run the test suite before committing:
  ```bash
  PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
  ```
- **Security Boundaries**:
  - Do not add `--dangerously-skip-permissions` to default arguments.
  - Preserve isolated HOME directory sandboxing and token symlink safeguards.
- **Async Compatibility**: Ensure the facade returns awaitable completions and async-iterable streams (`HERMES_SKIP_ASYNC_WRAP = True`).

---

## 3. Release and Upstream PR Protocol

Before cutting a release or updating the catalog pin, follow the protocol in [`.agents/skills/hermes-catalog-release/SKILL.md`](.agents/skills/hermes-catalog-release/SKILL.md).

### Summary Checklist
1. **Order of Operations**:
   - Update `README.md` to document new features, configuration, model compatibility, or caveats.
   - Update `version:` and `description:` in `plugin.yaml`.
   - Run tests (`pytest`).
   - Commit: `git commit -am "release: vX.Y.Z"`.
   - Push `main` to `origin/main` before creating the tag. CI verifies that the tag commit exists on `origin/main`.
   - Create and push tag `vX.Y.Z`.
   - Verify the published release with `gh release view vX.Y.Z`.
2. **Upstream Catalog PR**:
   - In `/usr/local/lib/hermes-agent`, branch from `main`: `catalog/bump-antigravity-pin-v<slug>`.
   - Update `plugin-catalog/antigravity-subscription-directsdk.yaml` with the 40-character commit SHA, version, and concise description (under 800 chars, no changelog bloat, single unified disclosure).
   - Validate catalog: `python3 scripts/validate_plugin_catalog.py plugin-catalog/antigravity-subscription-directsdk.yaml`.
   - Validate plugin: `hermes plugins validate /root/.hermes/plugins/antigravity-subscription-directsdk`.
   - Push to fork and open a PR against `NousResearch/hermes-agent:main`.
3. **PR Writing Style**:
   - English only.
   - Lead with facts and measurements.
   - Avoid buzzwords: "intelligent", "smart", "crucial", "game-changing", "seamlessly".
   - Avoid throat-clearing: "Here is what changed", "In order to".
   - No em dashes. Use colons or periods.
   - Active voice. Cite file paths, token limits, byte counts, and error signatures.
   - Qualified issue links: never use bare `#15` or `(#16)`. Use `soyelmismo/hermes-antigravity-subscription#15` so GitHub does not link to unrelated upstream issues.
