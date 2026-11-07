# Batch Local Counting

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit these files there; they are the only editable files, and no file may be added or
deleted: `counting/forest.h`, `counting/forest.cpp`, `counting/execution.h`,
`counting/execution.cpp`, `counting/aggregation.h`, `counting/aggregation.cpp`,
`executable/scope.cpp`, `executable/command.cpp`, `plan/tree.h`, `plan/tree.cpp`. Nothing
outside `/workspace` is writable.
