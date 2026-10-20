# In-place Integer Redistribution testbed

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read the research statement at [repo/README.md](repo/README.md), then the [platform guide](bin/README.md). Both are included in this public task tree.

Prepare an editable copy:

```sh
/frontier/task/bin/prepare_workspace.sh
cd /workspace/parallel_radix_sort
```

Only `RadixSort/edgelistgraph.h` and source headers under `RadixSort/frontier_solution/` belong in your patch. Build locally with `./frontier-build.sh` and run the authentic entrypoint with `./frontier-bench.sh`.

Create the payload and deterministic archive:

```sh
/frontier/task/bin/make_submission.sh /workspace/parallel_radix_sort /workspace/solution.patch
mkdir -p /workspace/submission/payload
cp /workspace/solution.patch /workspace/submission/payload/solution.patch
tar --format=ustar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf /workspace/answer.tar -C /workspace/submission payload/solution.patch
frontier-submit train /workspace/answer.tar
frontier-submit store-final /workspace/answer.tar
```

Submit an early validated implementation through train, and submit again after substantive progress to use verification feedback. After reviewing feedback, decide whether to replace your saved final. There is no minimum number of calls or fixed stopping schedule.

When finished:

```sh
frontier-submit final /workspace/answer.tar
```

Then exit. Formal answering lasts 18,000 seconds and includes train waits; final is independently measured with 900 seconds. Actual short trial time is supplied by the platform. You may make at most 100 train calls. Do not infer a remaining-time command that the installed Candidate does not provide.

Train feedback includes `dev_candidate_median_sort_time_ns` (the median of your
six public per-workload median sort times),
`dev_candidate_max_auxiliary_bytes` (your greatest auxiliary allocation in that
suite) and `dev_candidate_median_region_graph_ns` (the median of your six
per-workload region-graph component times). `beat_spec_progress` reports, for
each of `sort_time_ns`, `region_graph_ns` and `auxiliary_bytes`, aggregate
`passed`, `failed`, and `total` workload counts, plus `pass_percent` and
`fail_percent`. These counts describe the strict comparisons in the research
statement without identifying individual workloads or revealing reference
measurements. Final measurements and decisions remain private.

You must save the final answer you select with `frontier-submit store-final /workspace/answer.tar`.
You decide which answer to save and whether to replace that selection.
Confirm its accepted receipt; local edits and train feedback do not update saved bytes.
Saving does not evaluate, end answering, or lock final. There is no save-count quota;
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time limits still apply.
