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

On a GPU runner (see "Using a GPU" below), build and run the public benchmark with the task's
entrypoints; `frontier-bench.sh` runs the Wan entrypoint by default and HunyuanVideo with the
argument `hunyuan`:

```bash
G=/opt/frontier/candidate/bin/frontier-gpu
$G push gpurunner_... /workspace/repo /workspace
$G exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
$G exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh hunyuan
```

The runner image is the pinned Wan 2.1 + HunyuanVideo / PyTorch 2.6 CUDA 12.4 image with the
model weights offline. Public inputs and code are in `/workspace/repo`; the private
development and final suites are not on the Agent runner. Pull outputs you need below
`/workspace` with `frontier-gpu pull` before packaging a submission.

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
