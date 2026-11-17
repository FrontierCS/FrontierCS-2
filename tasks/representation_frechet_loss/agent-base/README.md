# Differentiable Population Statistics

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `frechet_distance/queue.py` and new or changed files below
`frechet_distance/frontier_solution/`, at most 20 changed files. Keep everything you write under `/workspace`.

On a GPU runner (see "Using a GPU" below), build with
`bash /workspace/repo/frontier-build.sh` and run the bench command
from the statement's "Running locally" section (a full warm start and one update); a bare
`frontier-bench.sh` runs the full default schedule, far beyond the GPU budget.

The runner image is `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with the task's packages and
the offline model snapshots below `$FRONTIER_MODEL_ROOT`. Public inputs and code are in
`/workspace/repo`; the private development and final suites are not on the Agent runner. Pull
outputs you need below `/workspace` with `frontier-gpu pull` before packaging a submission.
Your train calls run on the judge's A100-80GB and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type A100-80GB, at most 7200 s each and 7200 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 8,388,608 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
