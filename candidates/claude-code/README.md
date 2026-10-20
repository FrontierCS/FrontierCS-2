# Claude Code Candidate

Evaluated model: **Claude Opus 5.5** (`claude-opus-5-5`), effort `max`.

The overlay packages the official Linux/x64 Claude Code **2.1.296** native
package. The Dockerfile downloads the npm tarball, checks its SHA-512 and the
`--version` output, and copies the executable with its own dynamic loader and
shared libraries. It does not modify the executable.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-claude-code:2.1.296 candidates/claude-code
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

This harness calls the Anthropic API directly with the key rather than through
the Driver's model proxy. Supply the key only through the Run's secret
injection; never put it in Candidate JSON, images or receipts. The Agent's
network policy must allow `api.anthropic.com`.

`frontier-submit` and `frontier-gpu` (and their skills under `submit/` and
`gpu-helper/`) are the shared platform clients; the GPU client implements
`frontier.gpu-helper/v3`.

## Tests

```bash
cd candidates/claude-code && python3 -m pytest -q test_launcher.py
```
