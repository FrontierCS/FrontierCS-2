# Learned Transformer Layer Mapping

The task statement is at `/frontier/task/repo/README.md`; read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `dynamic_tanh.py` and new or changed files below `solver_extensions/` (at most 32 changed
files). Keep everything you write under `/workspace`.

On a GPU runner (see "Using a GPU" below):

```bash
G=/opt/frontier/candidate/bin/frontier-gpu
$G push gpurunner_... /workspace/repo /workspace
$G exec gpurunner_... -- bash /workspace/repo/frontier-build.sh
$G exec gpurunner_... -- python3 /workspace/repo/public_test.py
$G exec gpurunner_... -- bash /workspace/repo/frontier-bench.sh ARGS...
```

The runner has the public `flower_photos` photographs, extracted, at `/opt/frontier/public/flower_photos`. Pull
outputs you need below `/workspace` with `frontier-gpu pull` before packaging a submission.
Your train calls run on the judge's A100-80GB and report its metrics.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type A100-80GB, at most 900 s each and 1800 s in total for the whole run.
- **1024 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 8,388,608 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace` with the image's toolchain environment; they need
  no prefix.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
