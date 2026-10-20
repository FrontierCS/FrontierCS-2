# Qwen Code Candidate

Evaluated model: **Qwen 3.8 Max** (`Qwen/Qwen3.8-Max-VL-Thinking`, alias
`qwen3.8-max-vl-thinking`), effort `high`.

The overlay installs unmodified `@qwen-code/qwen-code` **0.25.0** from
`package-lock.json` with Node 22.23.3 and a private Node/libc runtime.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-qwen-code:0.25.0 candidates/qwen-code
```

## Launcher contract

The launcher writes fresh native settings for the `FRONTIER_CODEX_MODEL` alias
and the Run-local Driver Chat Completions endpoint, authenticated only with the
scoped `FRONTIER_MODEL_PROXY_TOKEN`. Provider keys stay in the LiteLLM gateway.

For `qwen3.8-max-vl-thinking` the launcher requests `reasoning_effort: high`
with `enable_thinking: true`, keeps the 1,000,000-token context, and sends
`max_tokens: null` through native `extra_body` so the CLI does not impose its
generated output cap. The `qwen3.8-max` alias keeps its older fixed wire
parameters.

## Tests

```bash
cd candidates/qwen-code && python3 -m pytest -q test_launcher.py
```

`offline-smoke.Dockerfile` runs `offline_container_smoke.mjs` against a built
overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`); set
`FIXTURE_QWEN_THINKING=1` to exercise the thinking alias.
