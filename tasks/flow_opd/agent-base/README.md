# OCR Distillation Trajectory Sampling

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workload and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`,
`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` and new or changed files below
`flow_grpo/frontier_solution/`, at most 32 changed files. In the two retained modules only the
bodies of `pipeline_with_logprob` and `sde_step_with_logprob` may change. Keep everything you write under `/workspace`.

On a GPU runner (see "Using a GPU" below), run the task's public build and benchmark
entrypoints from the pushed repository:

```bash
G=/opt/frontier/candidate/bin/frontier-gpu
$G exec gpurunner_... -- bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- bash /workspace/repo/frontier-bench.sh \
  --profile /workspace/repo/frontier_profile.json --output /workspace/bench-out
```
Your train calls run on the judge's H100 and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type H100, at most 3000 s each and 3000 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 536,870,912 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
