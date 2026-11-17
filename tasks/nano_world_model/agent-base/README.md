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
`src/diffusion/frontier_solution/`, at most 20 changed files. Keep everything you write under `/workspace`.

On a GPU runner (see "Using a GPU" below), run the task's public build check and, with the
environment listed under "Running locally" in the statement (checkpoint, clip manifest, dataset
and VAE paths; the runner has no Point Maze data), its benchmark:

```bash
G=/opt/frontier/candidate/bin/frontier-gpu
$G exec gpurunner_... -- bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- bash /workspace/repo/frontier-bench.sh
```
Your train calls run on the judge's A100-80GB and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type A100-80GB, at most 1800 s each and 1800 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 536,870,912 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
