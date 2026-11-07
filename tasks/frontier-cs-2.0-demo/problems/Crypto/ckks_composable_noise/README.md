# CKKS Precision Propagation at 2,048 Slots

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the `.go` files below `estimator/` there; the statement lists the fixed files that must
not change. The configuration manifest is `/frontier/task/repo/workload_manifest.json`. Go
1.23.12 and the Lattigo module cache are installed offline (`CKKS_MODULE_CACHE`);
`sh frontier-build.sh` builds the package and `sh frontier-bench.sh` runs the
multiply/relinearize/rescale program. Nothing outside `/workspace` is writable.
