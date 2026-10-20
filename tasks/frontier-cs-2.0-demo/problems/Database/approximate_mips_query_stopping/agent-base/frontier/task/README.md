# Approximate-MIPS Query Stopping

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read the [research statement](repo/README.md) for the scientific problem,
editable scope, workload and scoring. The [testbed guide](bin/README.md) covers
submission, feedback, saved final selection and resource limits.

The source is read-only at `/frontier/task/repo`. Prepare a writable copy:

```sh
/frontier/task/prepare_workspace.sh
# Edit only /workspace/repo/src/alg.cpp.
/frontier/task/make_submission.sh
```

The helper writes `/workspace/payload/solution.patch`; it does not submit or
create a tar. Train, store-final and explicit final all use the same archive:
an uncompressed POSIX tar containing `payload/` and exactly one nonempty regular
file, `payload/solution.patch`. No executable bit is required. The complete tar
must be at most 65,536 bytes, including tar headers and padding. Paths are
relative; links, devices, FIFOs, duplicate entries and parent traversal are
rejected. Only `src/alg.cpp` may change in the patch.

```sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/answer.tar payload
wc -c /workspace/answer.tar
frontier-submit train /workspace/answer.tar
frontier-submit store-final /workspace/answer.tar
# Continue improving; save again when choosing a replacement.
# When finished:
frontier-submit final /workspace/answer.tar
```

Use train for an early valid implementation and after substantive progress.
After feedback, decide whether to replace your saved final. No fixed submission
schedule or minimum number of calls is required. A successful explicit final
locks the answer; exit afterward. Its score is computed independently after
Agent cleanup and is not printed by the submission command.

Formal answering lasts 18,000 seconds (5 hours), including preparation and train
waits; independent final has a separate 900 seconds. There are 100 train calls.
Read the testbed guide and the actual Candidate-provided Run budget for details.

## Development feedback

Completed train feedback includes `score_unbounded` and the candidate-only aggregate diagnostics
`mean_recall`, `mean_kratio` (last-result inner-product ratio), `mean_time_ms`
(query milliseconds), and `mean_candidates` (distinct verified candidates).
These diagnostics are arithmetic means across the 12 evaluated workloads.
Invalid submissions can return zero with no candidate diagnostics.

`beat_spec_progress` reports `passed`, `failed`, `total`, `pass_percent`, and
`fail_percent` for each of `mean_query_time`, `recall_at_50`,
`last_result_inner_product_ratio`, and `verified_candidate_count`. Each metric
has `total=12`; percentages are rounded to two decimal places. All 48 comparisons
must pass. Reference measurements, margins, workload identities and individual
comparison rows stay private. Final feedback exposes no metrics to the solver.
