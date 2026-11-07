# KV Cache Compaction

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workload and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Everything under `compaction/` is editable, at most 40 changed files. Nothing outside
`/workspace` is writable.

On a GPU runner (see "Using a GPU" below), compact and answer public articles with your method:

```bash
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh 22073
```
