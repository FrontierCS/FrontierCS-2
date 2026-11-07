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
bodies of `pipeline_with_logprob` and `sde_step_with_logprob` may change. Nothing outside
`/workspace` is writable.

On a GPU runner (see "Using a GPU" below), run the task's public build and benchmark
entrypoints from the pushed repository:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh \
  --profile /workspace/repo/frontier_profile.json --output /workspace/bench-out
```
