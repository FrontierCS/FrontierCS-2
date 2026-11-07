# Exact Sequence Alignment

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `edlib/src/edlib.cpp` there; it is the only editable file, and only the reconstruction
function's body may change (see the statement). Nothing outside `/workspace` is writable.
