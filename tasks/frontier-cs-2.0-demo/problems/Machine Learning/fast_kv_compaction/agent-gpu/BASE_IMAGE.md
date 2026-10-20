# fast_kv_compaction: Agent GPU debug runner pin

The agent's runner uses the same base and Python closure as the Verifier, so
any result the agent measures on its runner carries over to grading. See
`../verifier-gpu/BASE_IMAGE.md` for the rationale, the sources and the
alternatives considered.

```dockerfile
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
# nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 platform manifest
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
```

## Shared with the Verifier

- **Python venv.** `/opt/frontier/python` (CPython 3.12) is installed from the
  same `requirements-cp312-linux-x86_64.lock`: 149 wheels with sha256 hashes,
  installed with `--no-deps --require-hashes --only-binary=:all:`. It covers
  `torch==2.8.0`, `transformers==4.57.1`, `vllm==0.11.0`,
  `accelerate==1.12.0` and `datasets==4.4.1`. Nothing is pip-installed outside
  the lock.
- **runtime-python.** `/opt/frontier/runtime-python` is the same shell
  wrapper: it exports `venv/bin` on PATH and execs the venv's python3.
- **Model.** The model is staged offline the same way: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
  goes to `/opt/frontier/models/Qwen3-4B`, read-only, with the files listed in
  `../model-assets.json`. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Differences from the Verifier

- **Contents.** There are no judge files, hidden articles, targets or verifier
  scripts. The image carries only the read-only upstream source tree at
  `/frontier/task/repo` and `task_runtime.py`, following the
  `sparse_video_attention_policy` agent-gpu pattern.
- **apt.** It adds debugging and build tools on top of the Verifier set
  (`bash ca-certificates gcc g++ git libnss-wrapper openssh-client
  openssh-server python3 python3-dev python3-venv tar`): `make`, `ripgrep`
  and `curl` are worth adding so the agent can build Triton/CUDA kernels.
  nvcc 12.8 already ships in the devel base.
- **Bootstrap and user.** It copies the trusted SSH bootstrap from
  `FRONTIER_GPU_SSH_BASE_IMAGE`, and task commands run as `frontier`
  (10001) with `HOME=TMPDIR=/workspace`.
