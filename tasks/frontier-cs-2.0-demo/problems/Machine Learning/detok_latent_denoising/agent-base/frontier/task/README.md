# Robust Image Tokenizer

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `models/detok.py`, `models/autoencoder.py`, `models/frontier_solution/` there, then:

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

A train verification may take up to 5,400 seconds and continues independently
after acceptance. Keep track of its verification ID and remaining answering
time. Query its status when you need feedback; do not submit a duplicate just
because verification is still queued or running.

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **6 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For a beat-spec task, `metrics` includes the approved candidate measurements and
`beat_spec_progress`: per-metric workload pass/fail counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.

The development metrics are `development_mean_ordinary_quantized_png_psnr_db`
and `development_mean_ema_quantized_png_psnr_db` (your mean reconstruction
quality in dB), and `development_mean_method_process_wall_seconds` (your mean
complete-process time in seconds). `beat_spec_progress` contains one entry for
each of `ordinary_quantized_png_psnr_db`, `ema_quantized_png_psnr_db`, and
`method_process_wall_seconds`. Each entry reports `passed`, `failed`, `total`,
`pass_percent`, and `fail_percent` across the development workloads. These are
aggregate counts and percentages; workload identities and reference values
remain private.

You must save the final answer you select. Build `/workspace/final.tar`
with the same layout, then store it:

```sh
frontier-submit store-final /workspace/final.tar
```

You decide which answer to save and whether to replace that selection. Each successful call replaces the
selected archive; editing local files does not refresh that selection. A failed
store preserves the previous selection. There is no save-count quota;
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time
limits still apply. Saving does not extend the answering window. Storing a selection
starts no verification and does not consume a train call or produce a score.

The answering window is 18,000 seconds. Independent final has a separate
5,400-second timeout and freshly evaluates the selected archive. An accepted
explicit final takes priority over the stored selection. On a platform-eligible
termination without explicit final, the Driver may use the latest stored
selection; cancellation or platform failure does not guarantee collection.
Neither stored nor accepted status establishes a measured final result.

When you finish, build `/workspace/final.tar` from the exact source you want
evaluated, refresh the stored selection, run

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
frontier-gpu create --type L4 --count 1 --seconds 300
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
Public inputs and code are in the
prepared `/workspace/repo`; private development/final suites are not available
on the Agent runner. Choose commands, remote working paths and output paths
freely within the task's contract. Pull needed outputs below `/workspace` with
`frontier-gpu pull` before packaging and using `frontier-submit train`.

GPU budget: 1 runner at a time on L4, 5,400s each and 5,400s in total for the whole run. Reuse a runner while useful, then delete it and confirm
termination. A timeout or lost create reply is not proof that allocation failed:
use list/status and the same printed idempotency key before retrying.

The Driver owns Modal credentials and allocation. Do not invoke Modal, inspect
the Driver bearer, call its endpoints directly, or replace the CONNECT proxy.
Runner experiments are debugging, not a score or a submission. After final
acceptance, exit immediately without further GPU operations.

## Testbed guide

See [testbed usage](bin/README.md) for packaging, checkpoint/final semantics,
and the existing policy limits.

## Preview answering window

The answering window is **5 hours (18,000 seconds)**. Independent final evaluation
starts after the answering window and has its own **5400-second** limit.
The existing train limits remain **6 calls** and **32400 cumulative seconds**.
Saving a final selection does not extend the answering window or change these limits.
