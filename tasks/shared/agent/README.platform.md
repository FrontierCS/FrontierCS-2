## Objective

1. **Pass first.** A submission passes when it beats the reference as defined in the
   statement's Scoring section under **Beats the reference**: on every final workload and every
   listed metric, with all correctness rules satisfied. Train `beat_spec_progress` shows, per
   metric, how many train workloads pass; final may use other or additional workloads and is judged on its own,
   so passing train does not guarantee passing final.
2. **Then raise the score.** Among passing submissions, a higher continuous score is better.
   Never replace a passing submission with a failing one to gain score. Until something passes,
   keep the best valid submission saved with `store-final`.

## Submitting

How the submission tool works (train, status, store-final, final) is in the
`/opt/frontier/candidate/skills/frontier-submit/SKILL.md` skill. This task's facts:

- Make the patch with `/frontier/task/make_submission.sh` (writes
  `/workspace/payload/solution.patch`). It applies the same scope check as the judge and
  refuses a patch that leaves this task's editable files, before you spend a train call.
- The patch may change at most {max_changed_files}.{file_rules}
- Train and final use one uncompressed POSIX tar holding exactly `payload/solution.patch`,
  {size_limits}; links, devices and FIFO members are rejected:

```sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
/opt/frontier/candidate/bin/frontier-submit train /workspace/train.tar
```

- Save your chosen answer early and update it with
  `/opt/frontier/candidate/bin/frontier-submit store-final /workspace/final.tar`; when done, run
  `/opt/frontier/candidate/bin/frontier-submit final /workspace/final.tar` exactly once and exit.
- A train submission is never chosen automatically as the final answer, and there is no fallback
  to the saved answer when the platform fails. If `store-final` is unavailable, report that
  rather than assume an answer was saved.

## Limits

- Agent machine: {cpus} CPU cores, {memory_gib} GiB memory, {workspace_gib} GiB writable
  `/workspace`.
- The judge evaluates your submission with {verifier_cpus} CPU cores and {verifier_memory_gib} GiB of
  memory.
- **{train_calls} train calls**; a call is consumed when the submit is accepted and is never
  refunded, even when verification or the infrastructure fails.
- Cumulative train wall time: **{train_wall_seconds} seconds**, counted from verifier start to
  result. One train may use all of the remaining train wall time (and no more than the remaining
  answering window); the {eval_timeout_seconds}-second per-evaluation timeout applies to final
  only. The wall time can run out before the train calls do.
- Only one train runs at a time: a pending train reserves all remaining train wall time, so a
  second submit is refused (HTTP 429 `quota_exhausted`, "verifier-train wall-time exhausted")
  and not charged a call; `frontier-submit` retries for about 100 s, then prints
  `Driver returned HTTP 429`.
- These limits are ceilings, not a promised number of completed evaluations.
- The overall answering window is **{wall_time_seconds} seconds**; Agent work and train share
  it.
- Final evaluation has a separate **{eval_timeout_seconds}-second** allowance; it does not
  consume train calls or the train wall-time budget. It runs after you exit, and the final
  command does not print its score.

## Feedback

A finished train result has `raw_score` (this submission's score on the train workloads) and
`metrics`; the evaluator's message (`summary`) is not returned. `metrics` holds:

- `score_unbounded`: the unbounded score.
- `beat_spec_progress`: per beat-spec metric, `passed`, `failed`, `total`, `pass_percent` and
  `fail_percent` (rounded to 2 decimals) over the train workloads. A pass is a reference
  comparison, not a correctness verdict.
{feedback}
- `rejected`, only for a submission the task's rules reject (before or during measurement): it
  scores 0, `metrics` holds only `score_unbounded` and `rejected`, and `rejected` is one of
  `empty`, `too_large`, `malformed`, `forbidden_file_type`, `out_of_scope`, `too_many_files` or
  `content_policy` (the same for every task). The train result does not list the files involved;
  `make_submission.sh` runs the same scope check first and names them.

A submission that fails to build or run, or fails the task's correctness checks, also scores 0.

Final metrics, per-workload results and reference measurements stay private.
