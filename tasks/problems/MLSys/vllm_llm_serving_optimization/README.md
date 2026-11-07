# vLLM serving and scheduling optimization

Read the statement, `/frontier/task/repo/README.md`. The packaged clean vLLM 0.11.0 checkout is
`/frontier/task/repo`; prepare your editable copy with

```sh
/frontier/task/bin/prepare_workspace.sh
```

which creates `/workspace/repo` (a git repository of the packaged source). Edit only the
allowlisted Python files the statement lists, then write your patch with
`/frontier/task/bin/make_submission.sh` (it writes `/workspace/payload/solution.patch`; an empty
patch evaluates unmodified vLLM). It diffs your working tree against the current `HEAD` of
`/workspace/repo`, so do not commit there: committed changes are left out of the patch. A server needs the H100 runner described below (the CPU
machine has no GPU). The runner holds the same vLLM 0.11.0 build and the model offline; run
commands there through `/opt/frontier/task-runtime COMMAND`, which restores its CUDA and Python
environment.
