# ZCode Candidate

Evaluated models: **GLM 5.3** and **GLM 5.3 Flash**, effort `max`.

The Dockerfile builds the unmodified official Z.ai ZCode CLI (`zcode-cli`
**0.16.9**) from `https://github.com/zai-org/ZCode` at commit
`29628c9acdb81b703bbd4080c207a0e7ce5e276e` with Node 24.14.0 and the frozen pnpm
lockfile. The overlay carries the bundled CLI, its licenses, a private Node
runtime, the shared Frontier clients and a thin headless launcher.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-zcode:0.16.9 candidates/zcode
```

## Launcher contract

The launcher writes the official provider configuration for the Run-local
Driver Chat Completions endpoint and its per-Run token; provider credentials
stay in the LiteLLM gateway. `fetch-sanitize.cjs` is a Node preload that removes
the SDK's own correlation headers only for that endpoint, because the Driver
rejects candidate-owned correlation headers.

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CODEX_MODEL` | LiteLLM model alias |
| `FRONTIER_CODEX_REASONING_EFFORT` | `low`, `high` or `max` (default `max`) |

The model context is declared as 1,048,576 tokens with a 262,144-token
per-response cap. The CLI runs once in `yolo` mode with `stream-json` events;
plugins, external MCP, web tools and project memory are disabled.

## Tests

```bash
cd candidates/zcode && python3 -m pytest -q test_launcher.py && node --test test-fetch-sanitize.cjs
```
