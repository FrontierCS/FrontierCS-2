# BBQ bitmap traversal

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## Answering time

Formal Agent answering is **18000 seconds (five hours)**, including bootstrap
and train waits. Final judging starts after Agent and train cleanup and has its
own **1800-second** budget; it does not consume the answering window.
Train-call, cumulative train-wall and GPU quotas are separate upper bounds,
not extra answering time.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `hardware/bbq/generator/bbq_level_lx.py`, `hardware/bbq/generator/solver_extensions/` there, then:

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
most 8,388,608 bytes. Member paths must be relative and must not contain
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
`metrics`. **100 train calls are available for the
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

The scalar development feedback fields are `dev_min_throughput_ops_per_second`
(the minimum candidate `throughput_ops_per_second` across the fixed public
development workloads, or 0 when any is invalid),
`dev_max_latency_picoseconds` (the maximum candidate worst latency in cycles
multiplied by its clock period, or -1 when any is invalid),
`dev_max_clock_period_picoseconds` (the maximum modelled clock period of the
candidate's public configurations, or -1 when any is invalid), the cycle-level
diagnostics `dev_min_throughput_ops_per_cycle` and `dev_max_latency_cycles`
(0 and -1 respectively when any workload is invalid), and `dev_valid_fraction` (the fraction of fixed public
development workloads that are fully correct). These fields describe only the
candidate's public measurements and do not reveal reference values or hidden
workload identities.

This task requires a Driver and Candidate supporting `store-final`. Save your
first usable chosen answer with `frontier-submit store-final /workspace/final.tar`,
then save again when you select a better revision. This saves one
replaceable snapshot; it does not evaluate it, consume a train call, or end your
work. A successful save replaces it; a rejected replacement preserves the last
accepted snapshot. Editing your local file alone does not update it. The save receipt reports the remaining Agent time; saving does not extend it. A save receipt is not a final score. An
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

## Testbed guide

See [testbed usage](bin/README.md) for packaging, checkpoint/final semantics,
and the existing policy limits.
