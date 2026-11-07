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
Nothing outside `/workspace` is writable.

On a GPU runner (see "Using a GPU" below), build and run the public benchmark with the task's
entrypoints; `frontier-bench.sh` runs the Wan entrypoint by default and HunyuanVideo with the
argument `hunyuan`:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh hunyuan
```

The runner image is the pinned Wan 2.1 + HunyuanVideo / PyTorch 2.6 CUDA 12.4 image with the
model weights offline. Public inputs and code are in `/workspace/repo`; the private
development and final suites are not on the Agent runner. Pull outputs you need below
`/workspace` with `frontier-gpu pull` before packaging a submission.
