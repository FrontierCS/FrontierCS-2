# Future Video Sampling

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workload and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `src/diffusion/df_sample.py`, `src/diffusion/gaussian_diffusion.py`,
`src/diffusion/flow_matching.py` and new or changed Python files (`.py`, `.pyi`) below
`src/diffusion/frontier_solution/`, at most 20 changed files. Nothing outside `/workspace` is
writable.

On a GPU runner (see "Using a GPU" below), run the task's public build check and, with the
environment listed under "Running locally" in the statement (checkpoint, clip manifest, dataset
and VAE paths; the runner has no Point Maze data), its benchmark:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
```

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
- The patch may change at most 20 files. It may not delete files.
- Train and final use one uncompressed POSIX tar holding exactly `payload/solution.patch`,
  at most 262,144 bytes; links, devices and FIFO members are rejected:

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

- Agent machine: 3 CPU cores, 12 GiB memory, 10 GiB writable
  `/workspace`.
- The judge evaluates your submission with 3 CPU cores and 12 GiB of
  memory.
- **3 train calls**; a call is consumed when the submit is accepted and is never
  refunded, even when verification or the infrastructure fails.
- Cumulative train wall time: **10800 seconds**, counted from verifier start to
  result. One train may use all of the remaining train wall time (and no more than the remaining
  answering window); the 10200-second per-evaluation timeout applies to final
  only. The wall time can run out before the train calls do.
- Only one train runs at a time: a pending train reserves all remaining train wall time, so a
  second submit is refused (HTTP 429 `quota_exhausted`, "verifier-train wall-time exhausted")
  and not charged a call; `frontier-submit` retries for about 100 s, then prints
  `Driver returned HTTP 429`.
- These limits are ceilings, not a promised number of completed evaluations.
- The overall answering window is **18000 seconds**; Agent work and train share
  it.
- Final evaluation has a separate **10200-second** allowance; it does not
  consume train calls or the train wall-time budget. It runs after you exit, and the final
  command does not print its score.

## Feedback

A finished train result has `raw_score` (this submission's score on the train workloads) and
`metrics`; the evaluator's message (`summary`) is not returned. `metrics` holds:

- `score_unbounded`: the unbounded score.
- `beat_spec_progress`: per beat-spec metric, `passed`, `failed`, `total`, `pass_percent` and
  `fail_percent` (rounded to 2 decimals) over the train workloads. A pass is a reference
  comparison, not a correctness verdict.
- `psnr`: Candidate mean PSNR of the predicted future frames over the train clips, in dB; higher is better.
- `ssim`: Candidate mean SSIM of the predicted future frames over the train clips; higher is better.
- `lpips`: Candidate mean VGG-LPIPS of the predicted future frames over the train clips; lower is better.
- `rejected`, only for a submission the task's rules reject (before or during measurement): it
  scores 0, `metrics` holds only `score_unbounded` and `rejected`, and `rejected` is one of
  `empty`, `too_large`, `malformed`, `forbidden_file_type`, `out_of_scope`, `too_many_files` or
  `content_policy` (the same for every task). The train result does not list the files involved;
  `make_submission.sh` runs the same scope check first and names them.

A submission that fails to build or run, or fails the task's correctness checks, also scores 0.

Final metrics, per-workload results and reference measurements stay private.

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`. This task's facts:

- Runners: 1 at a time, type A100-80GB, at most 1800 s each and
  1800 s in total for the whole run.
- **32 SSH sessions for Agent GPU debugging**; every accepted `exec`, `ssh`, `push` or
  `pull` consumes one, including a command that later fails. The Verifier has its own pools
  (36 sessions for train, 12 for final), separate from yours.
- At most 536,870,912 bytes per SSH session in each direction, including tar overhead.
- On the runner, prefix task commands with `/opt/frontier/task-runtime` (SSH is not a login
  shell); `/opt/frontier/runtime-python` is the runner image's Python.
- Wait until a runner is `ready` before using SSH. Sessions are not refunded, so batch work into
  one bounded command rather than many small ones. GPU work counts against the answering window.
