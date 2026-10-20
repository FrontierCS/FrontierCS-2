# Step Code Candidate (Step 5 Preview)

Evaluated model: **Step 5 Preview** (`stepfun/step-5-preview`), effort `high`.

Step Code (`https://github.com/stepfun-ai/Step-Code`, root package `stepcode`
**0.1.0**) publishes no npm package or release, so the Dockerfile builds the
repository at commit `d0ec996df81f4089054f2cc80f1ae11139761f81` with pnpm 9.15.9
and its frozen lockfile (`--ignore-scripts`, `build:offline`), then relinks
production dependencies only. The Step Code source is unmodified. Node 22.23.3
and its libraries travel in the overlay.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-stepcode:0.1.0 candidates/stepcode-step5
```

## Launcher contract

The launcher writes fresh `models.json` and `config.toml` under a private
per-run home. One custom `frontier` provider uses the `openai-completions` API
against the Run-local Driver Chat Completions endpoint with the
`FRONTIER_CODEX_MODEL` alias and `apiKey: $FRONTIER_MODEL_PROXY_TOKEN`.
Requests use OpenRouter's `reasoning: {"effort": "high"}` format; the launcher
selects `--thinking high`, 1,000,000 tokens of context and a 64,000-token output
cap. Step runs headless (`-p --mode json`) in its Full Access tier because the
other tiers refuse writes and commands in print mode.

`gateway-models.json` is the LiteLLM route used for the alias: it forwards to
`openrouter/stepfun/step-5-preview`, drops client output caps and
`reasoning_effort`, and injects `reasoning.effort: high`.

## Tests

```bash
cd candidates/stepcode-step5 && python3 -m pytest -q test_launcher.py test_litellm_route.py
```

`test_litellm_route.py` is opt-in: it skips unless `FRONTIER_TEST_LITELLM` is set
and the pinned LiteLLM package is installed. `offline-smoke.Dockerfile` runs `offline_container_smoke.mjs` against
a built overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`).
