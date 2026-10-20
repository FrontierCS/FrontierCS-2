# Muse Code Candidate

Evaluated models: **Muse Wonder 1.0** (effort `max`) and **Muse Spark 1.3**
(effort `high`).

The overlay packages Meta's official Muse Code **1.4.2** (`1.4.2-R4684.1`)
Linux x64 executable. The Dockerfile downloads it from Meta's public download
URL, checks its SHA-256, size and `--version`, and adds Node with its libraries
for the launcher.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-muse-code:1.4.2 candidates/muse-code
```

## Launcher contract

`launch` starts `launcher.mjs`, which runs the official host against a loopback
transport (`transport.mjs`). The transport serves exactly one row of
`model-catalog.json` (the row whose id equals the model alias) at
`GET /muse-code/models` and streams `POST /responses` to the Driver's
`/v1/responses` endpoint with the Agent-scoped Driver capability. It does not
rewrite prompts, tools, reasoning, usage or events, and does not retry.

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CODEX_MODEL` | `muse-wonder-1.0` or `muse-spark-1.3` |
| `FRONTIER_CODEX_REASONING_EFFORT` | Passed to `--reasoning-effort`; must be inside the model's documented range |

`gateway-models.yaml` is an example LiteLLM route for both aliases; it is not
part of the image. Adding a model changes the catalog and therefore requires a
new image.

## Tests

```bash
cd candidates/muse-code && node --test test_*.mjs
```

`test_official_host.mjs` runs against the real executable when
`MUSE_TEST_BINARY` names the downloaded binary (its SHA-256 is checked first);
otherwise it is skipped. `offline-e2e/` and `offline_container_smoke.mjs` are
loopback fixtures for a built overlay.
