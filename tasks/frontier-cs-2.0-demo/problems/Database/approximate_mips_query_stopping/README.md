# Approximate-MIPS Query Stopping

The task statement is at `/frontier/task/repo/README.md`.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/alg.cpp` there; it is the only editable file. Build with `make -j2` in
`/workspace/repo`. Nothing outside `/workspace` is writable.
