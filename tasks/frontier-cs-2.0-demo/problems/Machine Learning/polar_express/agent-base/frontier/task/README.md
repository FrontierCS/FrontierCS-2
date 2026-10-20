# Polar-Factor Approximation

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

Edit `polar_express.py`, `frontier_solution/` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

## Answering and verification time

The formal answering window is 18,000 seconds (five hours). Independent final
verification has its own 1800-second timeout and does not consume the answering
window. Train retains 100 calls and the original 24,000-second aggregate verifier
wall-time cap; this cap does not extend the answering window. Finish preparing
and selecting your answer before the answering deadline.

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

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **100 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For a beat-spec task, `metrics` includes the approved candidate measurements and
`beat_spec_progress`: per-metric workload pass/fail counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.

The candidate-only metrics are `worst_case_scalar_error`,
`relative_polar_factor_error`, `paper_regime_relative_polar_factor_error`,
`spectral_polar_error`, `polar_cosine_similarity` and
`top_subspace_polar_error`, each an arithmetic mean over the workloads that the
task statement names for it. `beat_spec_progress` has one entry for each of the
five criterion metrics (`worst_case_scalar_error`, `relative_polar_factor_error`,
`spectral_polar_error`, `polar_cosine_similarity`, `top_subspace_polar_error`),
each containing `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`.
The counts refer to workloads passing or failing that metric's strict
comparison; `total` is their sum, and the percentages use that total as the
denominator. Final metrics are withheld.

You must save the final answer you select with
`frontier-submit store-final /workspace/final.tar`. You decide which answer
to save and whether to replace that selection. This saves one
replaceable snapshot; it does not evaluate it, consume a train call, or end your
work. A successful save replaces it; a rejected replacement preserves the last
accepted snapshot. Editing your local file alone does not update it. There is no save-count quota; the receipt reports
`remaining_checkpoint_submissions` as `null`. Payload, audit and Agent-time
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

## Testbed guide

See [testbed usage](bin/README.md) for packaging, checkpoint/final semantics,
and the existing policy limits.
