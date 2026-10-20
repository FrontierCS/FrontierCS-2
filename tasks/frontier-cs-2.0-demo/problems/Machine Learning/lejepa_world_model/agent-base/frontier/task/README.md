# Latent Objective Recovery

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `experiments/lejepa_id/losses.py`, `experiments/lejepa_id/frontier_solution/` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

## Train, stored final and final tar format

Train, store-final and final use the same uncompressed POSIX tar format. The archive must
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

`frontier-submit train` returns immediately with state (`queued` or `running`),
`verification_id`, `payload_id`, and remaining train calls. Read the score and
task-approved development `metrics` later with
`frontier-submit status <verification_id>`. **8 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For this task, `metrics` includes scalar means of `r2_zh`, `r2_hz`,
`orth_err_normalized`, `epsilon`, `delta`, and `procrustes_mse` over development
workloads and `beat_spec_progress`.
For each metric, progress contains `passed`, `failed`, `total`, `pass_percent`,
and `fail_percent`; these are anonymous workload counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.
A train submission rejected because a loss carries state between calls scores
zero and its `metrics` holds only `candidate_diagnostic`: `category`
(`stateful_loss`) and the fixed `message` of that category.

The answering budget is 18000 seconds (5 hours). The maximum cumulative train
wall time is 3200 seconds across the 8 train calls. Independent final evaluation
has its own 400-second timeout after answering ends.

You must save the final answer you select with `frontier-submit store-final /workspace/final.tar`.
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
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it.
