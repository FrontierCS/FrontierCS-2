# DeepSeek Harness Candidate

Evaluated model: **DeepSeek V4.1 Flash**, effort `max`.

The overlay installs `@deepseek-ai/dsh` **0.2.0-rc.2** from `package-lock.json`
with `npm ci`, using its official `headless` profile and bundled `llm-pi-ai`
adapter. Node and its libraries travel in the overlay with a private ELF
interpreter. `patch_transport_error.mjs` makes one checked change to the locked
adapter: truncated streaming responses are classified as retryable transport
errors instead of invalid requests.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-deepseek-harness:0.2.0-rc.2 candidates/deepseek-harness
```

## Launcher contract

The adapter sends Chat Completions to the Run-local Driver model proxy. The CLI
receives only `FRONTIER_MODEL_PROXY_TOKEN`; provider credentials stay in the
LiteLLM gateway. Native account, search and telemetry features are disabled.

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CODEX_MODEL` | LiteLLM model alias |
| `FRONTIER_MODEL_CONTEXT_TOKENS` | Context window declared to the harness (1048576 for V4.1 Flash) |
| `FRONTIER_MODEL_OUTPUT_TOKENS` | Per-request output cap (262144 was used) |
| `FRONTIER_MODEL_REASONING_EFFORT` | `low`, `high`, `xhigh` or `max` |

The harness scrubs credential-shaped variables from shell subprocesses, so
`platform_tool.py` is installed as thin `frontier-submit` / `frontier-gpu`
wrappers that restore only the Agent's Driver token mount before running the
shared clients installed under `deepseek-harness/tools/`. The launcher also
creates a private `TMPDIR` inside the workspace because Agent root filesystems
are read-only.

## Tests

```bash
cd candidates/deepseek-harness && python3 -m pytest -q test_launcher.py
```

`test_transport_error.mjs` checks the patch against the installed adapter:
`node test_transport_error.mjs node_modules/@deepseek-ai/dsh-llm-pi-ai/lib/index.js`
after `npm ci`. `offline-smoke.Dockerfile` runs `offline_container_smoke.mjs`
against a built overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`).
