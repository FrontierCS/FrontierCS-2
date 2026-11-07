# Go1 Joystick Policy Training

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are new or changed files below `frontier_solution/` and
`mujoco_playground/_src/locomotion/go1/joystick.py`,
`mujoco_playground/_src/locomotion/go1/randomize.py` and
`mujoco_playground/config/locomotion_params.py`, at most 50 changed files. Nothing outside `/workspace` is writable.

On a GPU runner (see "Using a GPU" below), use the task's public check, `frontier_check.py`,
described in the statement. The runner image is `nvidia/cuda:12.6.3-base-ubuntu24.04` with
Python 3.12 and the hash-locked environment in `/frontier/task/runtime/requirements-gpu.lock`.
Public inputs and
code are in `/workspace/repo`; the private development and final suites are not on the Agent
runner. Pull outputs you need below `/workspace` with `frontier-gpu pull` before packaging a
submission.
