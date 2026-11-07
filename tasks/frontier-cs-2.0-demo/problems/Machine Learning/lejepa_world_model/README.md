# LeJEPA World Model

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `experiments/lejepa_id/losses.py` and `experiments/lejepa_id/frontier_solution/` there;
nothing else in the tree may change. Nothing outside `/workspace` is writable.
