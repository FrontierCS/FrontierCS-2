# Access-Pattern Rewriting

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Implement `code/src/language/solver_extensions/access_analysis.rs` there. You may also change
`code/src/language/rewrites.rs` (outside its fixed mapping code) and add Rust files under
`code/src/language/solver_extensions/`; every other file is fixed. Nothing outside `/workspace`
is writable. Rust 1.85.1 with the offline Cargo dependencies is installed; `frontier-build.sh`
and `frontier-bench.sh` in the repo build and test locally (see the statement).
