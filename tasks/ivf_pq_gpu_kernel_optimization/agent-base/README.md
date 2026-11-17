# GPU IVF-PQ Kernel Optimization

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workloads and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the Python files below `ivfpqlib/` there (new modules allowed). `make_submission.sh`
refuses a patch that changes a starter file outside the editable surface (revert it first); new
files outside it are left out of the patch. Keep everything you write under `/workspace`.

On a GPU runner (see "Using a GPU" below), build and run the task's checks:

```sh
G=/opt/frontier/candidate/bin/frontier-gpu
$G push gpurunner_... /workspace/repo /workspace
$G exec gpurunner_... -- bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- bash /workspace/repo/frontier-bench.sh
```

`frontier-build.sh` checks imports; `frontier-bench.sh [workload_id]` runs one public workload (default: the first).

The runner image is the pinned PyTorch 2.6 / CUDA 12.4 image with h5py, the SIFT-128 data at `/data/sift.hdf5` and the public benchmark at `/opt/frontier/development`; the hidden workloads and the optimized reference are not on it.
Your train calls run on the judge's H100 and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type H100, at most 10800 s each and 10800 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 67,108,864 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
