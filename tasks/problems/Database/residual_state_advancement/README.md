# Residual State Advancement

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `src/joinboost/app.py`, `src/joinboost/semiring.py`, `src/joinboost/aggregator.py`,
`src/joinboost/cjt.py` and `src/joinboost/joingraph.py` there; they are the only editable files.
The sandbox has no network: `pip install -e .` uses the setuptools wheel in `/opt/frontier/wheels`
(`PIP_FIND_LINKS`). Nothing outside `/workspace` is writable.
