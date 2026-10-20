# Exact DBSCAN Connectivity testbed

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read [the research statement](PROBLEM.md) for the algorithm, editable surface,
correctness, workloads and scoring. Read [the platform guide](bin/README.md)
for workspace setup, train, saved final and explicit final.

Prepare `/workspace/parallel_dbscan` with:

```sh
/frontier/task/bin/prepare_workspace.sh
```

Task files under `/frontier/task` are read-only. Edit only the authorized source
in your writable copy. The CPU testbed has Python 3.11, g++, OpenMP, make and git.

Train and final both submit an uncompressed tar containing the single regular
file `payload/solution.patch` (an optional `payload/` directory is allowed).
No executable bit is required. Patch limit: 262144 bytes and 20 changed files;
tar limit: 286720 bytes. There are 100 train calls. After creating the archive:

```sh
frontier-submit train /workspace/submission.tar
frontier-submit store-final /workspace/submission.tar
frontier-submit final /workspace/submission.tar
```

Use final once when selecting your completed answer, then exit. The bin guide
explains saved-copy replacement, termination eligibility and the independent
final clock. Formal answering is 18000 seconds; independent final is 600 seconds.

For a valid train submission, feedback includes
`candidate_elapsed_seconds_median` and `candidate_elapsed_seconds_p90`, the
candidate-only timing summaries defined in the research statement. It also
includes `beat_spec_progress.elapsed_seconds`: `passed`, `failed`, and `total`
count the strict per-workload timing comparisons; `pass_percent` and
`fail_percent` give their percentages. These aggregate counts reveal no workload
identities or reference timings. Invalid submissions score zero without timing
or progress metrics. Final returns no public metrics; its comparison results
remain private.

You must save the final answer you select with `frontier-submit store-final /workspace/submission.tar`.
You decide which answer to save and whether to replace that selection.
Confirm its accepted receipt; local edits and train feedback do not update saved bytes.
Saving does not evaluate, end answering, or lock final. There is no save-count quota;
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time limits still apply.

`frontier-submit train` returns immediately after acceptance with its verification ID
and remaining train-call count. Continue editing while verification runs;
use `frontier-submit status <verification_id>` to inspect feedback.
