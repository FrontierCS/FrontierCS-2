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
`frechet_distance/frontier_solution/`, at most 20 changed files. Nothing outside `/workspace`
is writable.

On a GPU runner (see "Using a GPU" below), build with
`/opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh` and run the bench command
from the statement's "Running locally" section (a full warm start and one update); a bare
`frontier-bench.sh` runs the full default schedule, far beyond the GPU budget.

The runner image is `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with the task's packages and
the offline model snapshots below `$FRONTIER_MODEL_ROOT`. Public inputs and code are in
`/workspace/repo`; the private development and final suites are not on the Agent runner. Pull
outputs you need below `/workspace` with `frontier-gpu pull` before packaging a submission.
