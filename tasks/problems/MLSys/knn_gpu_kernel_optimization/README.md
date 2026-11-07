# GPU Brute-Force k-NN Kernel Optimization

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workloads and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the Python files below `knnlib/` there (new modules allowed); the patch must not be empty. `make_submission.sh`
refuses a patch that changes a starter file outside the editable surface (revert it first); new
files outside it are left out of the patch. Nothing outside `/workspace` is writable.

On a GPU runner (see "Using a GPU" below), build and run the task's checks:

```sh
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
```

`frontier-build.sh` checks imports; `frontier-bench.sh` runs a small ungraded search.

The runner image is the pinned PyTorch 2.6 / CUDA 12.4 image with the read-only source tree; the graded datasets, the hidden workloads and the optimized reference are not on it.
