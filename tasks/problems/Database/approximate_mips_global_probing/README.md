# FARGO paper-aligned query core

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/alg.cpp` there; it is the only editable file. `make && ./fg` builds and runs a public
API smoke example; `bash make_submission.sh` writes the `src/alg.cpp` diff to `solution.patch`
for a local check. Nothing outside `/workspace` is writable.
