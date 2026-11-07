# Exact DBSCAN Connectivity

The task statement is at `/frontier/task/repo/README.md`.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the reconstruction region of `code/DBSCAN.C` and add files under `code/frontier_solution/`.
Build and run in `/workspace/repo/code` with `make` and `./DBSCAN` as the statement shows. Nothing
outside `/workspace` is writable.
