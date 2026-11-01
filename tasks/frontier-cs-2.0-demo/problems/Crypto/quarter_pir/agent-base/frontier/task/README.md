# Private Set List Decoding

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Everything you may change is the package directory `quarter_pir/code/pir/` there
(edit its files or add `.go` files; Go source only). `./frontier-build.sh` builds the tree and checks the source policy;
`./frontier-run.sh [entries]` runs one complete local experiment. Then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

The answering window is 18,000 seconds, including development evaluations.
There are 8 train calls with a cumulative 48,000-second verification allowance.
Final verification has its own independent 6,000-second timeout. Agent and
Verifier each have 4 CPUs and 8 GiB of memory.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 1,048,576 bytes. Member paths must be relative and must not contain
`..`; links, devices and FIFO members are rejected.

```sh
cd /workspace/repo && /frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

`frontier-submit train` returns immediately with a queued/running state,
`verification_id`, `payload_id`, and the authoritative count of remaining train
calls. Read the score and task-approved development `metrics` later with
`frontier-submit status <verification_id>` while continuing your work.
**8 train calls are available for the
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

This task requires a Driver and Candidate supporting `store-final`. Save your
first usable answer with `frontier-submit store-final /workspace/final.tar`,
then save again when you select a better revision. This saves one
replaceable snapshot; it does not evaluate it, consume a train call, or end your
work. A successful save replaces it; a rejected replacement preserves the last
accepted snapshot. Editing your local file alone does not update it. When the
tool reports `remaining_runtime_s` and `remaining_checkpoint_submissions`, honor
the remaining runtime and any integer checkpoint allowance. A null
`remaining_checkpoint_submissions` means saves are not count-limited.
Saving does not extend your answering time. A save receipt is not a final score. An
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
