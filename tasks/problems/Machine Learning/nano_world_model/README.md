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
