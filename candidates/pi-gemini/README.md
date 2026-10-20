# Pi Candidate (Gemini)

Evaluated model: **Gemini 4 Argon**, effort `high`.

The overlay installs unmodified `@earendil-works/pi-coding-agent` **1.2.0**
from `package-lock.json` with Node 22.23.3 and a private Node/libc runtime.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-pi:1.2.0 candidates/pi-gemini
```

## Launcher contract

The launcher writes fresh `models.json` and `settings.json` into a private
per-run `PI_CODING_AGENT_DIR`. One custom provider uses Pi's native
`openai-completions` API against the Run-local Driver Chat Completions endpoint
with the `FRONTIER_CODEX_MODEL` alias. Its `apiKey` is the literal reference
`$FRONTIER_MODEL_PROXY_TOKEN`, resolved from the environment per request; no
token or provider key is written to disk.

The model entry maps Pi thinking levels to Gemini `low`/`medium`/`high` and the
launcher selects `--thinking high`. The LiteLLM route for the alias should use
LiteLLM's Gemini translator and set
`generationConfig.thinkingConfig.thinkingLevel: HIGH` explicitly, dropping the
client's `max_tokens`, `max_completion_tokens` and `reasoning_effort`. Tool-call
ids are replayed verbatim so Gemini thought signatures survive.

Pi runs headless with `--mode json`, `--offline`, `PI_OFFLINE=1`,
`PI_SKIP_VERSION_CHECK=1` and `PI_TELEMETRY=0`, with Pi's default `read`,
`bash`, `edit` and `write` tools. Extensions, packages and MCP are disabled.

## Tests

```bash
cd candidates/pi-gemini && python3 -m pytest -q test_launcher.py
```

`offline-smoke.Dockerfile` runs `offline_container_smoke.mjs` against a built
overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`).
