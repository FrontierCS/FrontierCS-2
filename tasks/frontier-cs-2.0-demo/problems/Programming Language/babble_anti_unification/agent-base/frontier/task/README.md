# Reusable Library Candidates

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## Answering time

Formal Agent answering is **18000 seconds (five hours)**, including bootstrap
and train waits. Final judging starts after Agent and train cleanup and has its
own **600-second** budget; it does not consume the answering window.
Train-call, cumulative train-wall and GPU quotas are separate upper bounds,
not extra answering time.

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read the research statement at [/frontier/task/repo/README.md](repo/README.md)
first for the problem, editable surface, workloads, scoring and verifier resources.
The complete platform workflow and budgets are in [bin/README.md](bin/README.md).

## Workspace and payload

The source tree `/frontier/task/repo` is read-only. Create your writable copy:

```sh
/frontier/task/bin/prepare_workspace.sh  # -> /workspace/repo
```

Edit `src/learn.rs`, `src/dfta.rs`, and supporting `src/learn_extensions/` files
within the statement's exact restrictions. Then prepare the patch and package it:

```sh
/frontier/task/bin/make_submission.sh  # -> /workspace/payload/solution.patch
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/answer.tar payload
wc -c /workspace/answer.tar
```

The helper includes new allowed support files and returns the payload path.
You create the tar separately. The entire uncompressed archive must be at most
262144 bytes, containing `payload/` and exactly one nonempty regular file,
`payload/solution.patch`. The patch needs no executable bit. Paths are relative;
`..`, links, devices and FIFOs are forbidden. Train, store-final and final use
the same archive format and limits.

## Train and selected final

Submit an early valid implementation for feedback and train again after
substantive progress; choose when feedback would help. Up to 100 train calls
are available, subject also to 1800 cumulative train wall seconds and the
answering deadline. Accepted calls consume quota even if verification fails.

```sh
frontier-submit train /workspace/answer.tar
frontier-submit store-final /workspace/answer.tar
```

Completed train feedback includes the continuous score, approved absolute development measurements,
aggregate comparison progress and remaining calls. Save your first usable chosen answer with `store-final`, then, after
feedback, decide whether to replace your saved final. Saving takes a snapshot, runs no verifier, consumes
no train call, and allows continued work. Local edits alone do not update it.

When done, explicitly submit your chosen final once:

```sh
frontier-submit final /workspace/answer.tar
```

An accepted explicit final is irreversible and overrides any saved selection.
Otherwise eligible termination can collect the saved copy as detailed in
[the testbed guide](bin/README.md). A train score is never promoted to final.
Independent final evaluation starts after Agent/train cleanup, with its own
600-second budget. The formal answering budget is 18000 seconds (five hours).

## Feedback fields

Train metrics include `score_unbounded`, the continuous score before clipping,
and the candidate's absolute `compression_ratio` and median `runtime_seconds`.
`beat_spec_progress` summarizes comparisons for each of those two metrics using
`passed`, `failed`, `total`, `pass_percent`, and `fail_percent`. The counts and
percentages cover the evaluated development workloads without identifying which
workload passed or failed. Reference measurements and workload-specific
comparison outcomes remain private. Final evaluation exposes no metrics to the
Agent.
