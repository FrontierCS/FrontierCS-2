---
name: frontier-gpu
description: Use when a Frontier task explicitly provides an Agent GPU image and GPU quota, and remote GPU execution materially helps solve or test the task.
---

# Frontier GPU runner

Read `/frontier/task/README.md` first. Use a GPU only when that README says the
task supports one and identifies the usable GPU type/budget, runtime and data.
The GPU is an ephemeral Modal runner owned by the
Driver; it is not a second persistent workspace.

Run `frontier-gpu quota` to see the Driver's SSH `limit`, `used` and
`remaining` before planning GPU debugging. This HTTP query opens no SSH
session. Each accepted `exec`, interactive `ssh`, `push` or `pull` uses
one session; one transport connection is not a reusable multi-command quota.
With `scope=agent`, development and final verification have separately
reserved pools and do not consume your debugging sessions. With
`scope=legacy_shared`, they still share the displayed budget; heed the
warning. An older Driver may not expose quota: treat it as unknown, not
unlimited. Reaching zero prevents new debugging SSH, not CPU editing or a
submission under the separated policy. GPU seconds, transfer bytes, aggregate
evidence bytes and train-call limits remain independently enforced.

Use the Task's stated transfer-byte budget. Tasks explicitly granting 512 MiB
allow at most 536870912 bytes per logical SSH session in each direction,
including archive overhead; stdout and stderr share the GPU-to-Agent budget.
Older TaskVersions may grant less. `frontier-gpu quota` reports session counts,
not transfer-byte limits. Keep transfers small and treat quota rejection as a
failed transfer; never assume a partial download is a complete file.

Do CPU-side editing, syntax checks and small tests in `/workspace` before
allocating GPU time. Then create only the smallest allowed runner:

```bash
frontier-gpu create --type T4 --count 1 --seconds 300
frontier-gpu status gpurunner_...
```

The CPU Agent and GPU runner do not share files. Transfer only task code or
public inputs below `/workspace`, then execute a bounded command:

```bash
frontier-gpu push gpurunner_... /workspace/gpu-work /tmp
frontier-gpu exec gpurunner_... -- nvidia-smi
frontier-gpu exec gpurunner_... -- python /tmp/gpu-work/test.py
frontier-gpu pull gpurunner_... /tmp/gpu-work/output /workspace/gpu-output
```

Local upload sources and download destinations stay below `/workspace`.
`push SOURCE REMOTE_DIRECTORY` preserves SOURCE's basename: the example above
creates `/tmp/gpu-work/test.py`. Likewise, the pull example creates
`/workspace/gpu-output/output/`. Neither operation replaces a destination tree;
use a fresh directory when stale build files would affect a test.
Remote paths may be any absolute path writable/readable by the unprivileged GPU
user. `push` and `pull` reject links, special files and unsafe archive members,
and enforce a transfer limit. The remote image must provide `tar`; the Agent
CPU image must provide `/usr/bin/ssh`. Use `frontier-gpu ssh gpurunner_...`
only when interactive diagnosis is genuinely useful; prefer `exec` because it
is reproducible and bounded.

GPU execution is testing or artifact generation, not a Frontier submission.
Pull the allowed outputs back, inspect them on the CPU Agent, create the exact
task-defined tar, and use `frontier-submit train`. Iterate only while its score
and the remaining GPU/train quota justify another run. Delete a runner as soon
as it is no longer useful:

```bash
frontier-gpu delete gpurunner_...
```

Check `status` until the runner reports `terminated`; DELETE acceptance alone
is not proof of cleanup. If create loses its reply, inspect `list`/`status`
before retrying, and reuse the printed `--idempotency-key` with identical
arguments. Cold image imports can take much longer than an ordinary RPC.
Use the Python/toolchain paths and explicit environment documented by the Task:
an SSH command is not a login shell and need not inherit the image's environment.

The Driver also cleans up on timeout, cancellation, failure and finalization.
After `frontier-submit final` is accepted, do not create, connect to or mutate
a GPU runner; exit immediately.

Never invoke Modal directly, inspect the Driver bearer, call Driver endpoints
with `curl`, open a public SSH endpoint, replace the CONNECT proxy, or transfer
hidden eval data or credentials. SSH commands and plaintext stdin/stdout are
captured as sensitive bounded evidence. Avoid printing secrets and avoid
streaming large datasets, checkpoints or build trees unless the Task budgets
explicitly allow them.
