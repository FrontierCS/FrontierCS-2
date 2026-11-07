# Private Set List Decoding

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the package `quarter_pir/code/pir/` there (its files, or new `.go` files); nothing else may
change. Go 1.19.13 and the module cache are installed offline. `./frontier-build.sh` builds the
tree and checks the source policy; `./frontier-run.sh [entries]` runs one complete local
experiment. Nothing outside `/workspace` is writable.
