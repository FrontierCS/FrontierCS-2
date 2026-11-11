# Claude Code Candidate

Evaluated model: **Claude Opus 5.5** (`claude-opus-5-5`), effort `max`, with
the 2.1.296 recipe (see `image_source_commit` in `config/candidates.json`).
This recipe adds Driver gateway mode and has no published image yet.

The overlay packages the official Linux/x64 Claude Code **2.1.321** native
package. The Dockerfile downloads the npm tarball, checks its SHA-512 and the
`--version` output, and copies the executable with its own dynamic loader and
shared libraries. It does not modify the executable.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-claude-code:2.1.321 candidates/claude-code
```

`source.json` records the platform source revision and the SHA-256 of every
file the Dockerfile copies. The published image is listed in
[`config/candidates.json`](../../config/candidates.json).

## Launcher contract

`launcher.py` is installed as `/opt/frontier/candidate/launch` and runs one
print-mode CLI process (`stream-json`, empty MCP configuration, WebSearch and
WebFetch disabled) against `/frontier/task/README.md`. It creates fresh
home/config/tmp directories under `/workspace` and clears conflicting provider
and proxy variables.

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CLAUDE_MODEL` | Claude model id, for example `claude-opus-5-5` |
| `FRONTIER_REASONING_EFFORT` | Optional `low`, `medium`, `high`, `xhigh` or `max`; passed as `--effort` |
| `ANTHROPIC_API_KEY` | Anthropic API key, injected by the operator at run time |

With `ANTHROPIC_API_KEY` set, the harness calls the Anthropic API directly
rather than through the Driver's model proxy. Supply the key only through the
Run's secret injection; never put it in Candidate JSON, images or receipts. The
Agent's network policy must allow `api.anthropic.com`.

Without a key, the launcher runs in Driver gateway mode: the Driver injects
`ANTHROPIC_BASE_URL` (a frozen-IP `http` URL equal to
`FRONTIER_MODEL_API_BASE_URL` without `/v1`), `ANTHROPIC_AUTH_TOKEN` (the Run
token) and `ANTHROPIC_MODEL` (the alias, equal to `FRONTIER_CODEX_MODEL`). Every
Claude Code model variable is set to that alias. `FRONTIER_MODEL_OUTPUT_TOKENS`,
`FRONTIER_MODEL_CONTEXT_TOKENS` and `FRONTIER_AUTO_COMPACT_WINDOW` optionally set
the output, context and auto-compact limits. Any other value fails closed.

`frontier-submit` and `frontier-gpu` (and their skills under `submit/` and
`gpu-helper/`) are the shared platform clients; the GPU client implements
`frontier.gpu-helper/v3`.

## Tests

```bash
cd candidates/claude-code && python3 -m pytest -q test_launcher.py
```
