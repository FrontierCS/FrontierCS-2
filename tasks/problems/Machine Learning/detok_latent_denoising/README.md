# Robust Image Tokenizer

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `models/detok.py`, `models/autoencoder.py` and new or changed files
below `models/frontier_solution/`, at most 20 changed files. Nothing outside `/workspace` is
writable.

On a GPU runner (see "Using a GPU" below), build the tree and run the retained reconstruction
entrypoint with the task's scripts:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
```

`frontier-bench.sh` needs `FRONTIER_DETOK_DATA`, `FRONTIER_DETOK_FID_STATS` and
`FRONTIER_DETOK_CHECKPOINT` set to your own inputs; see the script.

The runner image is CUDA 12.4 with Python 3.10 and PyTorch 2.6, the task's hash-locked
packages, and the public model assets offline below `$FRONTIER_MODEL_ROOT` and `$TORCH_HOME`.
Public inputs and code are in `/workspace/repo`; the private development and final suites are
not on the Agent runner. Pull outputs you need below `/workspace` with `frontier-gpu pull`
before packaging a submission.
