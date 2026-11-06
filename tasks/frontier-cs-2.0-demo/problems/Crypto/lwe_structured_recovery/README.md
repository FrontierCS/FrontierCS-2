# Structured-LWE Public Witness Recovery

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Your answer is the witness ledger `/workspace/repo/solution.json`, the only editable file.
Accumulate valid witnesses in it with the ledger helper:

```sh
python3 /workspace/repo/tools/add_solution.py INSTANCE_ID '1,0,-1'
```

Keep your solver outside the provided files: elsewhere in `/workspace`, or as new files in
`/workspace/repo` (new files are never submitted). A patch that changes any other provided
file is refused, so restore experiments on the copied checker with `git checkout` before
submitting. Nothing outside `/workspace` is writable.
