# Dynamic GPU Allocation

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `scheduler/shockwave.py`, `scheduler/shockwave_helper.py` and files under
`scheduler/solver_extensions/` there (at most 20 files). Nothing outside `/workspace` is
writable. The simulator runs on CPU; `frontier-build.sh` and `frontier-bench.sh` in the repo
build and run it locally (see the statement).
