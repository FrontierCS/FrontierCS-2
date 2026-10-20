# Kimi Code Candidate

Evaluated models: **Kimi K3** (effort `high`) and **Kimi K2.7** (effort `medium`).

The overlay installs the official `@moonshot-ai/kimi-code` **0.31.0** CLI from
npm and runs it in non-interactive prompt mode with its native Agent Skills
support. The pinned bundle caps foreground Bash at 300 seconds; the Dockerfile
widens that one constant to 3600 and fails if the upstream bundle no longer
contains exactly one match.

This recipe is the attested `frontier.gpu-helper/v3` Candidate: its image pin,
source receipt and registry file readback are in
[`config/kimi-candidate-image.json`](../../config/kimi-candidate-image.json),
and `scripts/task_selection.py::attested_agent_gpu_interfaces` verifies every
file listed there before selecting GPU tasks that require V3. Changing any file
here invalidates that attestation until a new image is published and read back.
The images actually used for K3 and K2.7 are listed in
[`config/candidates.json`](../../config/candidates.json) with notes on how
their sources differ from this directory.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-kimi-code:0.31.0 candidates/kimi-code
```

## Launcher contract

The launcher writes one ephemeral Kimi configuration under `/workspace`. Its
`openai` provider points to the Run-local Driver `/v1` endpoint with the
Candidate's LiteLLM model alias; no provider credential enters the overlay or
the Agent environment.

| Variable | Meaning |
| --- | --- |
| `FRONTIER_CODEX_MODEL` | LiteLLM model alias |
| `FRONTIER_KIMI_MAX_CONTEXT_TOKENS` | Optional context size (default 262144; 1048576 for K3) |
| `FRONTIER_KIMI_SUPPORT_EFFORTS` | Optional JSON array of native effort levels, e.g. `["low","high","max"]` for K3 |
| `FRONTIER_KIMI_DEFAULT_EFFORT` | Effort selected from that array, e.g. `high` |

`frontier-submit train` returns immediately after acceptance;
`frontier-submit status VERIFICATION_ID` reads one result. Kimi's skill adds
Bash-specific guidance on top of the shared submission policy.

## Tests

```bash
cd candidates/kimi-code && python3 -m pytest -q test_launcher.py
```

The repository tests `tests/test_kimi_launcher.py` and `tests/test_kimi_submit.py`
cover the same launcher and submission client.
