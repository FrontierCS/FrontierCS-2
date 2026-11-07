# Search Strategy for Concolic Testing

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/frontier_solution/` there; nothing else in the tree may change. Nothing outside
`/workspace` is writable.

In `/workspace/repo`, `./frontier-build.sh` builds the tester and instruments the development
program (GNU grep 2.2, whose source is at `/frontier/task/benchmark-src/grep-2.2`), and
`python3 run_local.py` runs your strategy on it under the judge's limits.
