# Fixed-Length Clustered Sparse Multiplication

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `CSR_FlengthCluster.h`, `BIN_FlengthCluster.h`, `hash_mult_flengthcluster.h`,
`hash_mult.h`, `cluster_utility.h`, `CSR_VlengthCluster.h`, `BIN.h` and `frontier_solution/`
there; everything else is source-locked. Nothing outside `/workspace` is writable.
