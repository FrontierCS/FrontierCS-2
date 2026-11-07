# Multi-turn Trajectory Batches

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `ragen/llm_agent/ctx_manager.py` and new or changed files below
`ragen/llm_agent/solver_extensions/`, at most 20 changed files. Nothing outside `/workspace` is
writable.

On a GPU runner (see "Using a GPU" below), build the tree and run the retained one-step PPO
entrypoint with the task's scripts:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
```

The runner image is `hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2` (PyTorch 2.6, vLLM 0.8.2) with
the task's packages and the offline Qwen2.5-0.5B-Instruct snapshot below `$FRONTIER_MODEL_ROOT`.
Public inputs and code are in `/workspace/repo`; the private development and final suites are
not on the Agent runner. Pull outputs you need below `/workspace` with `frontier-gpu pull`
before packaging a submission.
