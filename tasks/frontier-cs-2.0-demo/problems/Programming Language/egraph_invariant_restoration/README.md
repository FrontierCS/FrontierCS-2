# E-Graph Invariant Restoration

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit only the source paths listed in the statement there; `src/lib.rs` is source-locked and
must remain byte-identical. New files below `src/solver_extensions/` are included in the
patch. Nothing outside `/workspace` is writable. Local checks: `bash ./frontier-build.sh` and
`bash ./frontier-bench.sh` in `/workspace/repo`.
