# vLLM serving and scheduling optimization

Read the statement, `/frontier/task/repo/README.md`. The packaged clean vLLM 0.11.0 checkout is
`/frontier/task/repo`; prepare your editable copy with

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

which creates `/workspace/repo` (a git repository of the packaged source). Edit only the
allowlisted Python files the statement lists, then write your patch with
`/frontier/task/make_submission.sh` (it writes `/workspace/payload/solution.patch`; an empty
patch evaluates unmodified vLLM). A server needs the H100 runner described below (this machine
has no GPU). The runner holds the same vLLM 0.11.0 build and the model offline; run commands there
through `/opt/frontier/task-runtime COMMAND`, which restores its CUDA and Python environment.

{platform}

## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`, which is not on `PATH`. This task's facts:

- Runners: 1 at a time, type H100, at most 18000 s each and 18000 s in total for the whole run.
- **512 commands for Agent GPU debugging**; every accepted `exec`, `push` or `pull` consumes one,
  including a command that later fails. Your train calls are a grant of their own.
- At most 268,435,456 bytes of files per `push` or `pull`, which copy directories.
- On the runner, commands run in `/workspace`, which is the only place they may write.
- `create` waits until a runner is `ready`. Commands are not refunded, so batch work into one
  bounded command rather than many small ones. GPU work counts against the answering window.
