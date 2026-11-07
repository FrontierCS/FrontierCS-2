# Learned Transformer Layer Mapping

The task statement is at `/frontier/task/repo/README.md`; read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `dynamic_tanh.py` and new or changed files below `solver_extensions/` (at most 32 changed
files). Nothing outside `/workspace` is writable.

On a GPU runner (see "Using a GPU" below):

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime python3 /workspace/repo/public_test.py
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh ARGS...
```

The runner has the public `flower_photos` archive at `/opt/frontier/public/flower_photos`. Pull
outputs you need below `/workspace` with `frontier-gpu pull` before packaging a submission.
