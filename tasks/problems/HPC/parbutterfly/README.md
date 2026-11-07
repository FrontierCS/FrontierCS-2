# ParButterfly Exact Counting

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit only the seven butterfly headers listed in the statement there, and add any supporting
C++14 sources below `apps/hyper/solver_extensions/` (new files there are included in the
patch). Nothing outside `/workspace` is writable.

Local compile check:

```sh
bash /workspace/repo/frontier-build.sh /workspace/repo
```
