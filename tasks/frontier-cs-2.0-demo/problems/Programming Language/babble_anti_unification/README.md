# Reusable Library Candidates

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/learn.rs`, `src/dfta.rs` and supporting `src/learn_extensions/` files there, within
the statement's marked regions (new files below `src/learn_extensions/` are included in the
patch). Nothing outside `/workspace` is writable. The toolchain is Rust 1.64.0
(`RUSTUP_TOOLCHAIN=1.64.0`), the version the judge builds with.
