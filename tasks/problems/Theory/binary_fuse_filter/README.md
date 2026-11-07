# Binary Fuse Filter Construction and Lookup

The task statement is at `/frontier/task/repo/README.md`.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `include/` there and add header-only helpers under it. `make query` builds the original
entrypoint and `make test` the retained tests. Nothing outside `/workspace` is writable.
