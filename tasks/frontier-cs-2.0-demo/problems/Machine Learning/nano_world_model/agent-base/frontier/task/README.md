# Future Video Sampling

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/diffusion/df_sample.py`, `src/diffusion/gaussian_diffusion.py`, `src/diffusion/flow_matching.py`, `src/diffusion/frontier_solution/` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 262,144 bytes. Member paths must be relative and must not contain
`..`; links, devices and FIFO members are rejected.

```sh
cd /workspace/repo && /frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **3 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For a beat-spec task, `metrics` includes the approved candidate measurements and
`beat_spec_progress`, keyed by metric name. Each metric contains `passed` (the
number of workloads meeting its reference-beating criterion), `failed` (the
number not meeting it), and `total` (`passed + failed`). A pass is a reference
comparison, not a correctness verdict. `pass_percent` is `100 * passed / total`;
`fail_percent` is `100 * failed / total`. Both percentages range from 0 to 100
and are rounded to two decimal places. This feedback is train-only; final/eval
metrics remain private. It does not return the evaluator's message, exact
workload pass/fail identities, or private reference measurements.

You must save the final answer you select with
`frontier-submit store-final /workspace/final.tar`. You decide which answer
to save and whether to replace that selection. This saves one
replaceable snapshot; it does not evaluate it, consume a train call, or end your
work. A successful save replaces it; a rejected replacement preserves the last
accepted snapshot. Editing your local file alone does not update it. There is no save-count quota;
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time
limits still apply. Saving does not extend your answering time. A save receipt is not a final score. An
accepted explicit final always wins. Otherwise the Driver can collect the saved
answer on normal exit, answering timeout, or a confirmed Agent process crash,
but not on cancellation or platform failure. Train submissions are never chosen
automatically. Check each save receipt; a local archive or a train result is not
a stored selection. If this required command is unavailable, report the runtime
incompatibility; do not assume a saved selection exists.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it.

## Using a GPU

Your container is a CPU editor/control environment. Use the Candidate-owned
`frontier-gpu` tool and read its installed skill at
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` before allocating a runner.
Its executable is `/opt/frontier/candidate/bin/frontier-gpu`; task convenience
scripts can invoke that absolute path directly. Reuse this CLI for transport
and authentication instead of implementing another Driver client.
The CPU workspace and GPU runner do not share files.
Run `frontier-gpu quota` before debugging to inspect your SSH limit, used
sessions and remaining sessions. Each accepted exec, interactive SSH, push or
pull consumes one session; the quota query itself does not. Agent debugging,
development verification and final verification have separate reserved SSH
budgets in this TaskConfig. Exhausting your debugging sessions does not prevent
CPU work or submission; GPU seconds, train calls and evidence bytes remain limited.

```bash
frontier-gpu quota
frontier-gpu create --type A100-80GB --count 1 --seconds 300
# Use the returned runner_id below; status must report ready before SSH.
frontier-gpu status gpurunner_...
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- nvidia-smi
# Run the task's documented public check before deleting the runner.
frontier-gpu delete gpurunner_...
frontier-gpu status gpurunner_...  # confirm terminated
```

The example preserves the uploaded directory's basename: `/workspace/repo`
lands at `/workspace/repo`, not `/workspace/repo/repo`. Use the task's public
build/benchmark entrypoints. Prefix runtime commands with
`/opt/frontier/task-runtime` to restore the image's toolchain/model environment;
SSH is not a login shell. `/opt/frontier/runtime-python` is the image's Python.
For interactive diagnosis, use `frontier-gpu ssh gpurunner_...`; use `exec`
for scripted checks and individual commands.
Runtime base: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726. Public inputs and code are in the
prepared `/workspace/repo`; private development/final suites are not available
on the Agent runner. Choose commands, remote working paths and output paths
freely within the task's contract. Pull needed outputs below `/workspace` with
`frontier-gpu pull` before packaging and using `frontier-submit train`.

GPU budget: 1 runner at a time on A100-80GB, 1,800s each and 1,800s in total for the whole run. Reuse a runner while useful, then delete it and confirm
termination. A timeout or lost create reply is not proof that allocation failed:
use list/status and the same printed idempotency key before retrying.

The Driver owns Modal credentials and allocation. Do not invoke Modal, inspect
the Driver bearer, call its endpoints directly, or replace the CONNECT proxy.
Runner experiments are debugging, not a score or a submission. After final
acceptance, exit immediately without further GPU operations.

## Testbed guide

See [testbed usage](bin/README.md) for packaging, checkpoint/final semantics,
and the existing policy limits.
