# Closed-loop action sampling

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit the body of `conditional_sample` in
`diffusion_policy/policy/diffusion_transformer_lowdim_policy.py` there, and optionally add Python
helpers under `diffusion_policy/sampler_extensions/`; nothing else may change. Nothing outside
`/workspace` is writable. `python3` is CPython 3.10.20 with the judge's pinned packages
(torch 2.2.2+cpu, numpy 1.23.5, diffusers 0.11.1, ...), so the repository runs locally on CPU.
