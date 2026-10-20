# Go1 Joystick Policy Training

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `frontier_solution/`, `mujoco_playground/_src/locomotion/go1/joystick.py`,
`mujoco_playground/_src/locomotion/go1/randomize.py` and
`mujoco_playground/config/locomotion_params.py` there, then:

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
`beat_spec_progress`: per-metric workload pass/fail counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.

The candidate measurements are `development_mean_episode_reward`,
`development_mean_tracking_lin_vel`, `development_mean_tracking_ang_vel`,
`development_mean_energy`, and `development_mean_action_rate`. Each is the mean
over the development workloads of your own policy's value for that metric, as
defined in the statement's Scoring section; higher is better for all five. For
each metric, `beat_spec_progress` reports `passed`, `failed`, `total`,
`pass_percent`, and `fail_percent`, with percentages rounded to two decimal
places. These are aggregate counts, without workload identities or private
reference values. Each development verification trains your procedure and the
reference once on a freshly provisioned verifier GPU, so it takes on the order
of 20 to 40 minutes, and its numbers carry training noise.

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
frontier-gpu create --type H100 --count 1 --seconds 1800
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
check, `frontier_check.py`, described in the statement. Prefix runtime commands with
`/opt/frontier/task-runtime` to restore the image's toolchain environment;
SSH is not a login shell. `/opt/frontier/runtime-python` is the image's Python.
For interactive diagnosis, use `frontier-gpu ssh gpurunner_...`; use `exec`
for scripted checks and individual commands.
Runtime base: nvidia/cuda:12.6.3-base-ubuntu24.04 with Python 3.12 and the
hash-locked environment in `/frontier/task/runtime/requirements-gpu.lock`. Public inputs and code are in the
prepared `/workspace/repo`; private development/final suites are not available
on the Agent runner. Choose commands, remote working paths and output paths
freely within the task's contract. Pull needed outputs below `/workspace` with
`frontier-gpu pull` before packaging and using `frontier-submit train`.

GPU budget: 1 runner at a time on H100, 14,400s each and 14,400s in total for the whole run. Runner time is spent from allocation, not from your first command: a fresh runner pulls a multi-gigabyte image, and the first JAX/CUDA use and every new JIT compilation take tens of seconds to minutes, so the budget is not all training time. Reuse a runner while useful, then delete it and confirm
termination. A timeout or lost create reply is not proof that allocation failed:
use list/status and the same printed idempotency key before retrying.

The Driver owns Modal credentials and allocation. Do not invoke Modal, inspect
the Driver bearer, call its endpoints directly, or replace the CONNECT proxy.
Runner experiments are debugging, not a score or a submission. After final
acceptance, exit immediately without further GPU operations.

## Task time and transport limits

The answering window is 18,000 seconds. Each train verification has a
5,400-second limit, with three calls and 16,200 train seconds in total.
Independent final evaluation has its own 14,400-second limit. A save or
submission does not extend these limits. Honor the tool's returned counters.

SSH session budgets are 32 for Agent debugging, 12 for all train verifications,
and 4 for independent final. Accepted failed or ambiguous operations may
consume sessions; one pool cannot borrow from another.

## Testbed guide

See [testbed usage](bin/README.md) for packaging, checkpoint/final semantics,
and the existing policy limits.

## Preview answering window

The answering window is **5 hours (18,000 seconds)**. Independent final evaluation
starts after the answering window and has its own **14400-second** limit.
The train limits are **3 calls** and **16200 cumulative seconds**.
Saving a final selection does not extend the answering window or change these limits.
