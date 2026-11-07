# SQL Counterexample Search

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `sql_testing/environment.py`, `sql_testing/smt/formula.py`,
`sql_testing/smt/knowledgebase.py`, and `sql_testing/smt/search_extensions/` there; nothing
else in the tree may change. Nothing outside `/workspace` is writable. The source attribution
of the Polygon artifact is in `/frontier/task/repo/UPSTREAM_ATTRIBUTION.md`.
