# KAMI FP64 Block GEMM — trusted batched harness adaptation

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workloads and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the device region of `src/block_gemm/KAMI/1d/fp64/block_gemm_1d_double_mma.cu` and, if you
like, files below `src/block_gemm/KAMI/1d/fp64/solver_extensions/`. `make_submission.sh` refuses
a patch that changes a starter file outside the editable surface (revert it first); new files
outside it are left out of the patch. Nothing outside `/workspace` is writable.

The Agent CPU container has no GPU. On a GPU runner (see "Using a GPU" below), compile and run
the public debug workload:

```sh
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
frontier-gpu pull gpurunner_... /workspace/repo /workspace/gpu-results
```

The runner is one exact NVIDIA H100 80GB HBM3 with CUDA 12.8 and the read-only source tree;
neither suite is on it.
