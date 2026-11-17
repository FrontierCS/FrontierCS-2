## Feedback

A finished train call's `returned` holds `raw_score` (this submission's score on the train
workloads) and the metrics below, each by its name; the evaluator's message is not returned.

- `score_unbounded`: the unbounded score.
- `beat_spec_progress`: per beat-spec metric, `passed`, `failed`, `total`, `pass_percent` and
  `fail_percent` (rounded to 2 decimals) over the train workloads. A pass is a reference
  comparison, not a correctness verdict.
{metrics}
- `rejected`, only for a submission the task's rules reject (before or during measurement): it
  scores 0, `returned` holds only `raw_score`, `score_unbounded` and `rejected`, and `rejected`
  is one of `empty`, `too_large`, `malformed`, `forbidden_file_type`, `out_of_scope`,
  `too_many_files` or `content_policy` (the same for every task). The train result does not
  list the files involved; `make_submission.sh` runs the same scope check first and names them.

{zero}

Final metrics, per-workload results and reference measurements stay private.
