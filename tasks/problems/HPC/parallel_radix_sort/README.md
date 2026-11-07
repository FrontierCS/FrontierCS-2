# In-place Integer Redistribution

The task statement is at `/frontier/task/repo/README.md`.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `RadixSort/edgelistgraph.h` and add text headers under `RadixSort/frontier_solution/`. In
`/workspace/repo`, `./frontier-build.sh` builds the authentic entrypoint with the pinned GCC
7.5/Cilk Plus compiler (`./frontier-cxx`) and `./frontier-bench.sh` runs it on the four pinned
workers. Nothing outside `/workspace` is writable.
