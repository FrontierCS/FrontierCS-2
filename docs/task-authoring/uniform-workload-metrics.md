# Every workload measures every metric

**Rule.** Every workload of a task measures every metric that
`judge/beat_spec.json` declares, and the PASS gate checks every declared metric
on every workload. A beat spec lists one metric set, and that set applies to
each workload.

**Why.** When metrics are split across workloads, each metric is judged on only
part of the task. A submission can then move cost from a measured place into an
unmeasured one, and that cost is never counted. Legitimate techniques such as
lazy evaluation stay allowed: with one metric set, deferred work is measured
where it actually lands.

**Example (EVM State Checkpoints).** The Spectrum workloads measured only
throughput; the checkpoint micro-workload measured only per-call create/restore
time and bytes. A lazy restore won `restore_ns` on the micro-workload, and the
deferred work was never charged: that workload had no throughput metric, and the
Spectrum runs never took the lazy path. The fix measures all metrics on all
workloads; it does not redefine the metrics.

## What `make validate` enforces

`scripts/validate_task_layout.py` checks each beat spec:

- Every workload row the spec declares carries the full metric set. Rows are
  listed under any top-level key ending in `workloads`. A row's metrics are its
  own `metrics` field, else the matching `<prefix>metrics` list (for example
  `final_only_workloads` with `final_only_metrics`), else every declared metric.
- `workload_groups`, which assigned metric subsets to groups of workloads, is
  retired.
- The task's `verifier/beat_spec_feedback.py` must reject a
  `beat_spec_workload_passes` row that lacks a declared metric.

Tasks listed in `NONUNIFORM_WORKLOAD_METRICS_EXEMPT` produce a warning instead of
a failure until their own fixes land. Remove a task's entry when it complies;
the validator fails on a stale entry.

Changing a task's metric set changes its verdicts: re-measure the reference and
release a new TaskVersion.
