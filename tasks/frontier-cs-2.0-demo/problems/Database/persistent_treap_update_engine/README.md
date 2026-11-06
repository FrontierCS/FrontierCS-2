# Persistent Treap Update Engine

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `lib/treap/scheduler.hpp`, `lib/treap/search_insert.hpp` and
`lib/treap/search_delete.hpp` there; they are the only editable files. Nothing outside
`/workspace` is writable.
