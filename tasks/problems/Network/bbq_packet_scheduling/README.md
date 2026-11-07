# BBQ bitmap traversal

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `hardware/bbq/generator/bbq_level_lx.py` there, and add helper modules below
`hardware/bbq/generator/solver_extensions/`; nothing else may change. Verilator, Yosys
(YoWASP) and sv2v are installed; `python3 frontier_cpu/runner.py --workdir DIR --official` builds and
measures the public cases (see "Running locally" in the statement). Nothing outside `/workspace` is writable.

