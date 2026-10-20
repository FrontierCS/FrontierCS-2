# Join-Tree Rewriting

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Completed train feedback includes the continuous score and aggregate pass/fail
progress per PASS comparison; final evaluates independently and retains
per-query PASS decisions in operator-only evidence.

Read the research statement at `/frontier/task/repo/README.md` first. Its source
is `agent-base/frontier/task/repo/README.md`, beside the public code. Read the
[testbed guide](bin/README.md) for budgets and selected-final behavior.

The source at `/frontier/task/repo` is read-only. Prepare a writable checkout:

```sh
/frontier/task/bin/prepare_workspace.sh
```

Edit only `/workspace/repo/2phase_nsa/binary_plan/rewrite_policy.py`. Create the
patch payload, then package it separately:

```sh
/frontier/task/bin/make_submission.sh
# Prints /workspace/payload/solution.patch; does not create a tar or submit.
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

Train, store-final, and explicit final accept the same uncompressed POSIX tar:
`payload/` containing exactly the nonempty regular file `payload/solution.patch`.
No executable bit is required. The whole archive must be at most 65,536 bytes,
including tar headers and padding; the patch's separate 65,536-byte limit does
not increase that archive limit. Reject absolute/traversing paths, duplicate
members, links, devices, FIFOs, and extra files. Work only under `/workspace`.

You must save your first usable chosen answer with `frontier-submit store-final`
and confirm the accepted receipt while continuing to work. You choose which
answer to save; after train feedback, decide whether to replace that selection.
Train never automatically selects the best or latest submission as final.

Package the answer you choose with the same layout and save it:

```sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/final.tar payload
frontier-submit store-final /workspace/final.tar
```

A successful save keeps a snapshot and permits continued work. Use the receipt
to confirm which archive is selected. When ready to finish, explicitly finalize:

```sh
frontier-submit final /workspace/final.tar
```

An accepted explicit final is irreversible and takes priority over a saved
selection. It ends answering; the Driver cleans up and independently evaluates
the selected archive. Train results are never substituted for final evaluation.

There are 100 train calls and 24,000 aggregate train seconds. Completed train
feedback includes the score, the submission tool's remaining-call information,
and these `metrics` over the 20 train queries:

- `score_unbounded`: the continuous score.
- `beat_spec_progress`: one entry for `exact_result` and one for
  `median_latency`. Each contains `passed`, `failed` and `total` query counts and
  the percentages `pass_percent` and `fail_percent`.
- `all_metrics_progress`: the same five fields for queries that clear both
  comparisons. PASS on the train suite is `passed == total` here.
- `candidate_median_latency_us`: the median of your own per-query median
  latencies, in microseconds; `null` unless every train query was measured.

These are aggregates. Query identities, per-query decisions and latencies,
reference values, ratios and margins are not returned. Final returns no metrics.
