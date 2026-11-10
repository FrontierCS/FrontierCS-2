# Curve Fitting on Sandboxes (template)

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workloads and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the files below `model/` there (new modules allowed). Nothing outside `/workspace` is
writable. `python3 run_local.py` fits and scores your model on a curve like the judge's.

The judge runs your `fit` and `predict` on a GPU runner of the same image as yours (see
"Using a GPU" below), so try them there:

```sh
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime python3 /workspace/repo/run_local.py
```
