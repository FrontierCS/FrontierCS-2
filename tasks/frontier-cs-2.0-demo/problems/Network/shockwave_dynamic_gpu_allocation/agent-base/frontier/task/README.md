# Dynamic GPU Allocation

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

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `scheduler/shockwave.py`, `scheduler/shockwave_helper.py`, `scheduler/solver_extensions/` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit. Operational details, including saved-final
selection, are in `/frontier/task/bin/README.md`.

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

Each `beat_spec_progress` entry contains `passed`, `failed`, and `total` workload
counts, plus `pass_percent` and `fail_percent` percentages. The entries cover
`makespan_seconds`, `average_jct_seconds`, `worst_finish_time_fairness_rho`, and
`unfair_job_fraction_percent`. These same four names identify the candidate's
anonymous ordered measurement arrays in `metrics`.

Save your first usable chosen archive while continuing to work.
After train feedback, decide whether to replace that selection:

```sh
frontier-submit store-final /workspace/train.tar
```

A successful replacement supersedes the prior saved archive; a rejected save
preserves it. Saving neither trains nor ends answering. Local edits do not alter
the Driver-owned saved bytes. Rejected saves do not replace the last successful selection.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it. Explicit final has priority over a saved archive. Without an
explicit final, normal exit, answering timeout, or a confirmed process crash may
collect the saved archive. User cancellation and platform failure do not. A
train result is never selected automatically.
