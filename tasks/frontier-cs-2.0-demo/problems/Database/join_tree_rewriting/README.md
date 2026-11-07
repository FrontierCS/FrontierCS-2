# Join-Tree Rewriting

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `2phase_nsa/binary_plan/rewrite_policy.py` there; it is the only editable file. Nothing
outside `/workspace` is writable. The planner modules in `2phase_nsa/binary_plan/` run locally
(see the statement's "Running locally"); the join engine and the data are judge-only.
