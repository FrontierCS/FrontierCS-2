# Sparse Video Attention Policy

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `svg/models/wan/attention.py`, `svg/models/hyvideo/attention.py`,
`svg/kmeans_utils.py` and new or changed files below `svg/solver_extensions/`, at most 12 changed
files (see the statement for the rules on them). Keep generated outputs out of those paths.
Keep everything you write under `/workspace`.

The GPU runner has the same read-only starter and the offline Wan 2.1 and HunyuanVideo
checkpoints. After editing on the CPU, make your submission patch. Push the payload directory
containing only `solution.patch` to a leased runner and apply it to a fresh copy of its starter.
Choose a new directory for each changed patch and set `runner` to the ID printed by
`frontier-gpu create` (see "Using a GPU" below).

```bash
G=/opt/frontier/candidate/bin/frontier-gpu
trial=/workspace/bench-1
/frontier/task/make_submission.sh
$G push "$runner" /workspace/payload /workspace
$G exec "$runner" -- /frontier/task/prepare_workspace.sh "$trial"
$G exec "$runner" -- git -C "$trial" apply /workspace/payload/solution.patch
$G exec "$runner" -- /opt/frontier/task-runtime env FRONTIER_REPO_ROOT="$trial" bash "$trial/frontier-build.sh"
$G exec "$runner" -- /opt/frontier/task-runtime env FRONTIER_REPO_ROOT="$trial" bash "$trial/frontier-bench.sh" wan
```

Use `hunyuan` in place of `wan` for the HunyuanVideo benchmark. Both entrypoints write below
`$trial/result`. Pull the outputs you want to inspect. For the Wan example:

```bash
$G pull --into-new-directory "$runner" "$trial/result/wan/t2v/ear" /workspace/wan-result-1
find /workspace/wan-result-1 -type f
```

The private development and final suites run on the judges.

Your train calls run on the judge's four H100s and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type H100, at most 18000 s each and 54000 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 536,870,912 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace`; prefix them with `/opt/frontier/task-runtime` to
  restore the image's toolchain and model environment.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
