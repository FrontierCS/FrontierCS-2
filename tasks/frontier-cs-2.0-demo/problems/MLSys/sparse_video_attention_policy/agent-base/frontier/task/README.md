# Sparse Video Attention Policy

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/bin/prepare_workspace.sh          # -> /workspace/repo
```

Edit `svg/models/wan/attention.py`, `svg/models/hyvideo/attention.py`, `svg/kmeans_utils.py`, `svg/solver_extensions/` there, then:

```sh
/frontier/task/bin/make_submission.sh            # -> /workspace/payload/solution.patch
```

The helper stages tracked edits and new files, checks the original editable
paths and 12-file limit, and writes the patch. It does not package or submit it.
Only changes to `svg/models/wan/attention.py`, `svg/models/hyvideo/attention.py`,
`svg/kmeans_utils.py`, and `svg/solver_extensions/` are permitted. Keep generated outputs out of the
submission changes.

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit. Read [the testbed instructions](bin/README.md)
for saving a selected final and resource limits.

The formal Agent answering budget is **5 hours (18000 seconds)**. Independent
final evaluation has a separate **12600-second** budget after answering ends.
Train waits and GPU development count within answering time.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty; it is read as data and does not
need an executable bit. The `payload/` directory must be searchable. The complete archive must be at
most 262,144 bytes. Member paths must be relative and must not contain
`..`; links, devices and FIFO members are rejected.

```sh
cd /workspace/repo && /frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **25 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For a beat-spec task, `metrics` includes the approved candidate measurements and
`beat_spec_progress`: per-metric workload pass/fail counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.

You must save your first usable chosen answer with `frontier-submit store-final`
and confirm the accepted receipt while continuing to work. You choose which
answer to save; after train feedback, decide whether to replace that selection.
Train never automatically selects the best or latest submission as final.

Build `/workspace/final.tar` with the same layout and save your chosen answer:

```sh
frontier-submit store-final /workspace/final.tar
```

Wait for an accepted receipt. A saved selection lets you continue working;
a later accepted `store-final` replaces it. After train feedback, decide whether
to update that selection. Editing local `final.tar` alone does not update the
saved archive. Train results never automatically select a final.

When you are ready to finish, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it. If answering ends normally, times out, or the Agent crashes
without an accepted explicit final, the Driver collects the last accepted stored
archive for independent final evaluation. User cancellation and platform faults
do not trigger this fallback. Without either accepted form, there is no final
answer to evaluate.

## Using a GPU

Your container is a CPU editor/control environment. Use the Candidate-owned
`frontier-gpu` helper V3 (`frontier.gpu-helper/v3`) and read its installed skill at
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` before allocating a runner.
Its executable is `/opt/frontier/candidate/bin/frontier-gpu`; task convenience
scripts can invoke that absolute path directly. Reuse this CLI for transport
and authentication instead of implementing another Driver client.
The CPU workspace and GPU runner do not share files.

```bash
frontier-gpu quota
frontier-gpu create --type H100 --count 1 --seconds 18000
# Use the returned runner_id below; status must report ready before SSH.
frontier-gpu status gpurunner_...
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- nvidia-smi
# Build and run the public benchmark within this runner allocation.
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh hunyuan
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

You have **1024 SSH sessions for Agent GPU debugging**. Every accepted `exec`,
interactive `ssh`, `push`, or `pull` consumes one session, including an accepted
command that later fails. Sessions are not refunded, and opening another
connection does not reset the count. Reuse a runner and combine related work
into a bounded command when useful.

The Verifier has independent SSH pools: **50 sessions across the 25 train calls**
and **2 sessions reserved for final evaluation**. Agent debugging, train, and
final do not consume one another's SSH allowances. Exhausting your debugging
sessions prevents new debugging SSH commands; CPU editing and submissions can
continue within their own limits.

`frontier-gpu quota` shows your Agent SSH `limit`, `used`, and `remaining`, with
`scope=agent` and `verifier_uses_separate_budget=true` for this task. This query
opens no SSH session and consumes none of the allowance. It does not reveal
private Verifier usage. Each SSH session allows at most **512 MiB (536870912 bytes) per direction**,
including tar overhead;
stdout and stderr share the GPU-to-Agent limit. GPU seconds, transfer bytes,
whole-run audit bytes, wall time, and train calls remain separate limits.

Runtime base: the pinned Wan 2.1 + HunyuanVideo / PyTorch 2.6 CUDA 12.4 image with offline model assets.
`frontier-bench.sh` runs the Wan entrypoint by default and the HunyuanVideo
entrypoint with the argument `hunyuan`. Public inputs and code are in the
prepared `/workspace/repo`; private development/final suites are not available
on the Agent runner. Choose commands, remote working paths and output paths
freely within the task's contract. Pull needed outputs below `/workspace` with
`frontier-gpu pull` before packaging and using `frontier-submit train`.

GPU budget: 1 runner at a time on H100, 18,000s each and 54,000s in total for the whole run. Reuse a runner while useful, then delete it and confirm
termination. A timeout or lost create reply is not proof that allocation failed:
use list/status and the same printed idempotency key before retrying.

The Driver owns Modal credentials and allocation. Do not invoke Modal, inspect
the Driver bearer, call its endpoints directly, or replace the CONNECT proxy.
Runner experiments are debugging, not a score or a submission. After final
acceptance, exit immediately without further GPU operations.
