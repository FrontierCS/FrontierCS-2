# Codex Candidate (built in)

Evaluated models: **GPT-6 Astra** and **GPT-6.1 Sol**, effort `xhigh`.

Codex is the built-in Candidate: `make apply` mirrors the overlay pinned in
[`config/task-catalog.json`](../../config/task-catalog.json) and registers it.
This directory is the recipe for that overlay. It also holds the canonical
sources of the shared Frontier clients that every other harness copies:

- `frontier_submit.py` and `skills/frontier-submit/SKILL.md` (`frontier-submit`)
- `frontier_gpu.py`, `frontier_gpu_connect_proxy.py` and
  `skills/frontier-gpu/SKILL.md` (`frontier-gpu`, `frontier.gpu-helper/v3`)

The Dockerfile adds the official `@openai/codex` 0.162.1 Linux x64 package by
checksum, verifies the executable hash and `--version`, and installs the
launcher, clients and skills with read/execute-only modes.

## Build

This directory is its own build context:

```bash
docker build --platform linux/amd64 -t frontier-candidate-codex:0.162.1 candidates/codex
```

## Launcher contract

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CODEX_MODEL` | Model alias accepted by the provider or LiteLLM gateway |
| `FRONTIER_CODEX_REASONING_EFFORT` | Optional `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` (`qwen3.8-max-vl-thinking`: `none` or `high`, default `high`) |
| `FRONTIER_CODEX_CONTEXT_WINDOW` | Optional context window (16384 to 2147483647) for a model outside Codex's bundled catalog |
| `FRONTIER_MODEL_DIRECT_BASE_URL` | Optional HTTPS OpenAI-compatible endpoint for direct mode |
| `CODEX_API_KEY` | Provider key, direct mode only |

A model outside Codex's bundled catalog gets its fallback metadata: a 272,000-token
context window and `reasoning.summary="auto"`. With `FRONTIER_CODEX_CONTEXT_WINDOW`
the launcher writes a one-row native `model_catalog_json` for the alias with that
window. The row reuses the fallback instructions byte for byte from the
checksum-pinned `fallback-base-instructions.md` and requests no reasoning summary;
without the variable the command is unchanged.

Without a direct endpoint the launcher uses the Run-local Driver model proxy and
the child receives no provider key. The launcher sets `allow_login_shell=false`
so shell commands keep the Candidate's tool PATH.

## Tests

```bash
cd candidates/codex && python3 -m pytest -q test_*.py
```

`test_official_cli_*.py` additionally exercise the real CLI when
`FRONTIER_TEST_CODEX_BINARY` names the pinned executable; otherwise they skip.
