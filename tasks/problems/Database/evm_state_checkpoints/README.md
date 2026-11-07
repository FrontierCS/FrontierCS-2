# EVM State Checkpoints

The task statement is at `/frontier/task/repo/README.md`. Read it first: it defines the
problem, the workload, and how you are scored. This file covers only where things are and how
to submit.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `lib/evmcow/execution_state.hpp` there; it is the only editable file. Nothing outside
`/workspace` is writable.

Spectrum's CMake project fetches pinned dependencies, and the sandbox has no network: they are
prebuilt read-only at `/opt/frontier/spectrum-deps`. Copy them into your build tree before
configuring (the build tree is never submitted):

```sh
cd /workspace/repo && mkdir -p build && cp -a /opt/frontier/spectrum-deps build/_deps \
  && chmod -R u+rwX build
```
