# Register-aware Branch Prediction

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the nine editable regions of `my_cond_branch_predictor.cc` there (the statement lists
them) and add helper files under `solver_extensions/`. `make` builds the simulator `./cbp`;
`bash make_submission.sh` writes the predictor diff to `solution.patch` for a local check.
Nothing outside `/workspace` is writable.
