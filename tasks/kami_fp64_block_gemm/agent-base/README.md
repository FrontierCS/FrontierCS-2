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
outside it are left out of the patch. Keep everything you write under `/workspace`.

The Agent CPU container has no GPU, but it holds the CUDA 12.8 toolchain, so `frontier-build.sh`
also compiles the device-only cubin and the trusted host here, without running a workload. On a GPU
runner (see "Using a GPU" below), compile and run the public debug workload:

```sh
G=/opt/frontier/candidate/bin/frontier-gpu
$G push gpurunner_... /workspace/repo /workspace
$G exec gpurunner_... -- bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- bash /workspace/repo/frontier-bench.sh
$G pull gpurunner_... /workspace/repo /workspace/gpu-results
```

The runner is one exact NVIDIA H100 80GB HBM3 with CUDA 12.8 and the read-only source tree;
neither suite is on it. Your train calls run on the judge's H100 and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type H100, at most 900 s each and 900 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 536,870,912 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
