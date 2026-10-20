# Residual State Advancement

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

Edit `src/joinboost/app.py`, `src/joinboost/semiring.py`, `src/joinboost/aggregator.py`, `src/joinboost/cjt.py`, `src/joinboost/joingraph.py` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit. Operational details, budgets, and selected-
final behavior are in `/frontier/task/bin/README.md`.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 131,072 bytes. Member paths must be relative and must not contain
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
once verification completes, it includes your score and the task-approved unbounded score and development `metrics`. **100 train calls are
available for the whole run**, and every accepted request consumes one even when
verification fails. Private reference measurements and the final reference
verdict are not returned to the Agent.

Train feedback includes `metrics.score_unbounded` and one
`metrics.beat_spec_progress` entry for each beats-reference metric:
`elapsed_seconds`, `residual_update_seconds` and `rmse_error`. Each progress
object contains `passed`, `failed`, and `total` workload counts, plus
`pass_percent` and `fail_percent` percentages. These aggregate fields report
the per-workload comparisons across the evaluated workloads; they do not reveal
workload identities or candidate/reference timings. Final evaluation returns no
solver-visible metrics.

You may preserve a chosen archive while continuing to work:

```sh
frontier-submit store-final /workspace/train.tar
```

Saving has no save-count quota. The save receipt reports
`remaining_checkpoint_submissions` as `null`. A successful replacement supersedes the prior
saved archive; a rejected save preserves it. Saving neither trains nor ends answering.
Local edits do not alter the Driver-owned saved bytes.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it. An accepted explicit final is irreversible and has priority
over a saved archive. Without an explicit final, normal exit, answering timeout,
or a process crash confirmed by trusted container state may collect the saved
archive. User cancellation and platform failure do not. Without either selected
archive there is no final; a train result is never selected automatically.

The formal answering clock is 18,000 seconds and includes bootstrap and train
waits. After Agent and train cleanup, final evaluation receives its independent 7,200
seconds. A separately published short trial uses 1,800 answering seconds; its
Agent-visible instructions must identify that actual release budget.
