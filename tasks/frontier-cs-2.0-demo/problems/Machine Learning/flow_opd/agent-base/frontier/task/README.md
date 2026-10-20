# OCR Distillation Trajectory Sampling

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`, `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`, `flow_grpo/frontier_solution/` there, then:

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
/opt/frontier/candidate/bin/frontier-submit train /workspace/train.tar
```

Use the absolute Candidate helper path shown here: a fresh login shell can reset
`PATH`, and environment changes in a previous tool call do not configure it.

`/opt/frontier/candidate/bin/frontier-submit train` returns immediately with
`state` (`queued` or `running`), `verification_id`, `payload_id`, and the
authoritative count of remaining train calls. Read the score and task-approved
development `metrics` later with
`/opt/frontier/candidate/bin/frontier-submit status <verification_id>`. **4 train calls are available for the
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

You must save the final answer you select with `/opt/frontier/candidate/bin/frontier-submit store-final /workspace/final.tar`.
You decide which answer to save and whether to replace that selection.
Confirm the accepted receipt; local edits and train results do not update
the saved selection or automatically select a final. Saving has no save-count
quota; `remaining_checkpoint_submissions` is `null`. Payload, audit and
Agent-time limits still apply. Saving does not extend answering time.

A successful later save replaces the selection; a rejected save preserves it.
Saving does not evaluate the answer, consume a train call or end your work.
An accepted explicit final takes priority. Without an explicit final, the
Driver evaluates the last accepted stored answer after normal exit, answering
timeout or confirmed Agent process crash, but not cancellation or platform
failure. Without either accepted form, there is no final answer to evaluate.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
/opt/frontier/candidate/bin/frontier-submit final /workspace/final.tar
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
frontier-gpu create --type H100 --count 1 --seconds 300
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
Runtime base: Driver-approved GPU SSH base + Task dependencies. Public inputs and code are in the
prepared `/workspace/repo`; private development/final suites are not available
on the Agent runner. Choose commands, remote working paths and output paths
freely within the task's contract. Pull needed outputs below `/workspace` with
`frontier-gpu pull` before packaging and using
`/opt/frontier/candidate/bin/frontier-submit train`.

GPU budget: 1 runner at a time on H100, 3,000s each and 3,000s in total for the whole run. Reuse a runner while useful, then delete it and confirm
termination. A timeout or lost create reply is not proof that allocation failed:
use list/status and the same printed idempotency key before retrying.

The Driver owns Modal credentials and allocation. Do not invoke Modal, inspect
the Driver bearer, call its endpoints directly, or replace the CONNECT proxy.
Runner experiments are debugging, not a score or a submission. After final
acceptance, exit immediately without further GPU operations.

## Run limits

The formal answering budget is 18000 seconds (five hours). The 4 train calls
share a 12,000-second aggregate verification budget. Independent final evaluation
has its own 3000-second timeout after the Agent exits and does not consume the
answering window.

The formal Agent answering budget is **5 hours (18000 seconds)**.
Independent final evaluation has a separate budget after answering ends.
