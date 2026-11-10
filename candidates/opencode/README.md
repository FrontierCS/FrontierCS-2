# OpenCode Candidate

Not yet evaluated: no overlay image of this recipe is published.

The overlay installs unmodified OpenCode **1.19.5** (`opencode-ai`) from its npm
lockfile and takes the `opencode-linux-x64` CLI the wrapper selects on glibc
x86-64. The binary embeds its own JavaScript runtime and needs no Node; its
glibc loader and libraries are private. It ships ripgrep 15.1.0 for OpenCode's
`grep`/`glob` tools. No provider credential is part of the overlay.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-opencode:1.19.5 candidates/opencode
```

## Model route

The launcher writes a fresh `opencode.json` (`OPENCODE_CONFIG`) into a
private per-run home under the workspace. One custom provider, `frontier`,
uses the bundled `@ai-sdk/openai-compatible` SDK against the run-local Driver
Chat Completions endpoint (`FRONTIER_MODEL_API_BASE_URL`, `/v1`) with the
explicit `FRONTIER_CODEX_MODEL` LiteLLM alias. Its `apiKey` is the literal
`{env:FRONTIER_MODEL_PROXY_TOKEN}`, which OpenCode substitutes from the
process environment when it loads the config; the token is not written to
disk. `enabled_providers` admits only `frontier`.

The model declares reasoning and tool calls, a context and output limit of
262,144 tokens and an input limit
of 196,608 tokens. The upstream API can report fewer prompt tokens than it counts once it renders
the history's reasoning, so compaction starts with a quarter of the window to
spare.
`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` raises OpenCode's default
32,000-token per-response cap to that limit, so every request carries
`max_tokens: 262144`; the upstream API clamps it to the remaining window.
The explicit input limit keeps auto-compaction at its normal threshold
(input limit minus OpenCode's 20,000-token reserve). Without it, OpenCode
would subtract the whole output limit from the context and compact every
turn. Reasoning effort and temperature come from the gateway alias; the
launcher selects no variant. Streamed `reasoning_content` becomes OpenCode
reasoning parts, and the SDK replays it on later assistant messages as
`reasoning_content`.

## Runtime behavior

The launcher runs `opencode run --format json` (JSONL events on stdout) in
`/workspace` with the task prompt on stdin. `opencode run` reads stdin
verbatim, while message words in argv would be re-quoted. A fixed session
title skips OpenCode's separate title-generation request. `--auto` approves
any permission request instead of auto-rejecting it, so no prompt blocks.
The config also allows every permission (`*`, external directories,
doom-loop), then denies `webfetch` and `websearch`. `opencode run` itself
denies `question` and plan mode. Tools are OpenCode's defaults for this
model: `bash`, `read`, `edit`, `write`, `glob`, `grep`, `task`
(subagents), `todowrite` and `skill`. No MCP servers, plugins, LSP servers or
formatters are configured. Snapshots are off. The two Frontier skills load
from `skills.paths`.

Offline and isolation switches, all checked in the 1.19.5 source:
`OPENCODE_DISABLE_AUTOUPDATE`, `OPENCODE_DISABLE_MODELS_FETCH` (bundled
models.dev snapshot), `OPENCODE_DISABLE_LSP_DOWNLOAD`, `OPENCODE_PURE` (no
external plugins), `OPENCODE_DISABLE_DEFAULT_PLUGINS` (no third-party provider
auth plugins), `OPENCODE_DISABLE_PROJECT_CONFIG` (workspace `opencode.json`,
`.opencode/` and `AGENTS.md` are ignored), `OPENCODE_DISABLE_CLAUDE_CODE` and
`OPENCODE_DISABLE_EXTERNAL_SKILLS`. Session sharing is disabled. Telemetry is
off unless `OTEL_*` is set, and the launcher strips `OTEL_*`. ripgrep is on
`PATH`, so OpenCode does not download it. At startup OpenCode installs
`@opencode-ai/plugin` from npm into its writable global config directory
unless that directory already has `node_modules` and a lockfile listing the
package. The launcher seeds that marker. Plugins are disabled, so the package
is never loaded.

Inherited `OPENCODE_*`, `XDG_*`, `OTEL_*`, provider and `*_API_KEY` variables
are removed. `HOME`, the XDG directories and `TMPDIR` point into
`/workspace/.frontier-opencode-home`, which also holds the SQLite session store
and `data/opencode/log/opencode.log`.

OpenCode has no session time limit. The Driver owns the Agent deadline.
Provider requests have no total timeout, and the response-header and
stream-chunk timeouts are one hour. OpenCode retries retryable provider
errors (5xx, 429, network) five times, honouring `Retry-After` and otherwise
backing off exponentially from 2 s; this is not configurable. After that the
run ends. Bash commands time out
after OpenCode's default 2 minutes unless the model passes a longer `timeout`.
Automatic compaction stays enabled. Tool output above 2,000 lines or 50 KiB is
truncated, with the full text saved to disk.

Train returns asynchronous acceptance. Saving is not final submission: launcher
success requires the common client's accepted final receipt. A later CLI error
does not revoke an already accepted final.

Usage is counted by the trusted gateway callback. OpenCode's `step_finish`
token counts are diagnostic only.

## Tests

```bash
cd candidates/opencode && python3 -m pytest -q test_launcher.py
```

`offline-smoke.Dockerfile` runs `offline_container_smoke.mjs` against a built
overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`): the real launcher
and CLI, unprivileged, read-only and network-isolated, against dummy model and
Driver servers.
