---
name: frontier-gpu
description: Use when a Frontier task explicitly provides an Agent GPU image and GPU quota, and remote GPU execution materially helps solve or test the task.
---

# Frontier GPU runner

Read `/frontier/task/README.md` first. Use a GPU only when that README says the
task supports one and identifies the usable GPU type/budget, runtime and data.
The GPU is a runner that the host leases to you, with no network; it is not a
second persistent workspace. The helper is
`/opt/frontier/candidate/bin/frontier-gpu`.

Run `frontier-gpu quota` to see the `limit`, `used` and `remaining` of the
grant's commands and transfers and of its GPU seconds before planning GPU
debugging. Each accepted `exec`, `push` or `pull` uses one command, never
refunded, including a command that later fails. `create`, `status`, `list`,
`delete` and `quota` use none. A runner's seconds count from when it is ready
until it is deleted or its time runs out, whether or not it runs a command.
Reaching zero prevents new GPU commands, not CPU editing or a submission.
Train calls are a grant of their own, and use neither.

Use the Task's stated transfer-byte budget: each `push` or `pull` moves at most
`max_transfer_bytes` of files (`frontier-gpu quota` shows it). Keep transfers
small and treat a refused transfer as a failed one; never assume a partial
download is complete.

Do CPU-side editing, syntax checks and small tests in `/workspace` before
allocating GPU time. Then create a runner of the task's type:

```bash
frontier-gpu create --type H100 --count 1 --seconds 300
frontier-gpu status gpurunner_...
```

`create` prints the runner's ID and waits until the runner is ready. Starting
can take minutes while the host finds a GPU: `status` says `starting` and for
how long, which costs no GPU seconds. A runner that can't start is `failed`,
with the provider's reason: that is not your fault and costs no GPU seconds;
create another. The runner lasts `--seconds` from when it is ready, at most the
task's runner seconds.

The CPU Agent and GPU runner do not share files. Transfer only task code or
public inputs below `/workspace`, then execute a bounded command:

```bash
frontier-gpu push gpurunner_... /workspace/gpu-work /tmp
frontier-gpu exec gpurunner_... -- nvidia-smi
frontier-gpu exec gpurunner_... -- python3 /tmp/gpu-work/test.py
frontier-gpu pull gpurunner_... /tmp/gpu-work/output /workspace/gpu-output
```

Local upload sources and download destinations stay below `/workspace`.
`push SOURCE REMOTE_DIRECTORY` preserves SOURCE's basename: the example above
creates `/tmp/gpu-work/test.py`. Likewise, the pull example creates
`/workspace/gpu-output/output/`. `push` and `pull` copy directories, not
single files. `push` replaces the copy of SOURCE on the runner whole, so a
file that the last push or a command left there and SOURCE lacks is gone;
`pull` copies over what DESTINATION already holds. Each file and directory
keeps its permission bits, but not a set-ID or sticky bit, and a directory
arrives with at least `rwx` for its owner. `push` rejects links and special
files.
The runner keeps its files between commands, until it is deleted or its time
runs out. Commands run in `/workspace` on the runner, with the image's
toolchain environment; they need no prefix.

`exec` waits for the runner to be ready and idle, runs one command, waits for
it, prints the last 21 KiB of its stdout and stderr, and exits with its exit
status: 255 if the runner ended under it.
The whole output of command N stays on the runner in
`/pumpkin/log/N/command.stdout` and `command.stderr`; `exec` names them when
it showed only their end, and `pull` copies them back. One command or transfer
runs at a time. If `exec` is interrupted, the command goes on to its end, and
the next one waits for it; `delete` ends it. There is no interactive
`frontier-gpu ssh`: use `exec`, which is reproducible and bounded.

GPU execution is testing or artifact generation, not a Frontier submission.
Pull the allowed outputs back, inspect them on the CPU Agent, create the exact
task-defined tar, and use `sandboxctl call train`. Iterate only while its
score and the remaining GPU/train quota justify another run. Delete a runner as
soon as it is no longer useful:

```bash
frontier-gpu delete gpurunner_...
```

The runner's state lives on the host: `status` and `list` read it there, and
deleting `/workspace/.frontier-gpu/runners.json` loses nothing. The host also
ends every runner at your time's end and at finalization. After
`sandboxctl call final ... --arg finalize=true` is accepted, do not run GPU
commands; exit immediately. Avoid printing secrets and avoid streaming large
datasets, checkpoints or build trees.
