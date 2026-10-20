# fast_kv_compaction: Verifier GPU runtime pin

Checked from registry and index metadata only. No image layers
were pulled and no packages were installed.

## Recommended base

```dockerfile
# nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 platform manifest
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
```

- The multi-arch index digest for the same tag is
  `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66`.
  Pin the amd64 manifest above so the build cannot pick another platform.
- The image ships CUDA 12.8.1 with nvcc, NCCL 2.25.1 and Ubuntu 24.04. It was
  created on 2025-03-10 and its compressed layers total 5.16 GB.
- Python: Ubuntu 24.04's `python3` is CPython 3.12. Upstream needs
  Python 3.12, and the wheel lock below targets exactly CPython 3.12,
  Ubuntu 24.04 and x86_64.
- The image does not need cuDNN. torch 2.8.0 brings its own cuDNN wheel
  (`nvidia-cudnn-cu12==9.10.2.21`) and the other CUDA 12.8 runtime wheels.

`judge/config.yaml`: `gpu_image: nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43`

## Python: a venv installed from the hashed 149-wheel lock

Upstream (`adamzweiger/compaction@2020f6d`, `requirements.txt`) needs Python 3.12
and these exact versions: `torch==2.8.0`, `transformers==4.57.1`, `vllm==0.11.0`,
`accelerate==1.12.0` and `datasets==4.4.1`.

Reuse the closed lock from the authoring runtime admission (stage287), stored at
`agent-base/frontier/task/runtime/requirements-cp312-linux-x86_64.lock`.

- The lock has 149 packages with one `--hash=sha256:` per package. Every
  dependency is listed, so the install runs with `--no-deps --require-hashes`.
  The wheels total 4.99 GB.
- I checked all 149 `name==version` + sha256 pairs against the
  PyPI JSON API. All 149 resolve to an official, non-yanked wheel, with
  0 mismatches.
- It contains these direct pins: `torch==2.8.0`, `transformers==4.57.1`,
  `vllm==0.11.0`, `accelerate==1.12.0`, `datasets==4.4.1`,
  `huggingface-hub==0.34.4`, `numpy==2.2.6`, `pyyaml==6.0.2` and
  `zstandard==0.25.0`. Selected transitive packages: `torchvision==0.23.0`,
  `torchaudio==2.8.0`, `xformers==0.0.32.post1`, `triton==3.4.0`,
  `tokenizers==0.22.1`, `safetensors==0.8.0`, `xgrammar==0.1.25`,
  `ray==2.48.0`, `numba==0.61.2`, `cupy-cuda12x==13.6.0`, and
  `nvidia-*-cu12` 12.8.x (cuBLAS 12.8.4.1, NCCL 2.27.3, cuDNN 9.10.2.21).
- NumPy is 2.2.6, not 1.x: `pyarrow==26.0.0` (pulled in by `datasets`)
  refuses to import with NumPy below 2.0, which failed the first H100 smoke
  (`import datasets`). Every locked package's PyPI `requires_dist` admits
  2.2.6; the tightest bounds are `numba==0.61.2` (`<2.3`) and
  `opencv-python-headless==4.12.0.88` (`>=2,<2.3`).
- Two conflicts are already fixed in the lock:
  - `opencv-python-headless==4.12.0.88`, the release that matches NumPy 2
    (4.11 was only kept while the lock froze NumPy 1.26.4). vLLM asks only
    for `>=4.11.0`.
  - `setuptools==79.0.1`, because vLLM 0.11.0 requires `>=77.0.3,<80` on
    Python above 3.11.
- vLLM 0.11.0 pins torch exactly. Its PyPI `requires_dist` includes
  `torch==2.8.0`, `torchaudio==2.8.0`, `torchvision==0.23.0`,
  `xformers==0.0.32.post1`, `transformers>=4.55.2` and
  `setuptools<80,>=77.0.3` (python>3.11). The lock agrees with all of them.
  The vLLM wheel hash
  `3861c75ff2b12e24f6d179ff5c084d791b42ded8675d76c8706697c79f68cd62` matches
  PyPI's `vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl`.
- Status of the lock: the stage283/287 audit shows the metadata graph is
  closed (149 nodes, 0 unresolved, 0 version conflicts). It is not a runtime
  qualification. The lock has never been installed, imported or run on a GPU.
  The first build plus a smoke test (`import vllm, torch`, plus one Qwen3-4B
  generate) is the real check.

Recipe (the same one as the reviewed stage287 V3, with the base swapped
from the internal gpu-ssh-base to the public digest above):

```dockerfile
ARG DEBIAN_FRONTEND=noninteractive
COPY agent-base/frontier/task/runtime/requirements-cp312-linux-x86_64.lock /opt/frontier/python-lock/requirements.lock
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates gcc g++ git libnss-wrapper \
       openssh-client openssh-server python3 python3-dev python3-pip python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --only-binary=:all: --require-hashes --ignore-installed \
       --index-url https://pypi.org/simple -r /opt/frontier/python-lock/requirements.lock \
    && /opt/frontier/python/bin/python -m pip check
ENV PATH="/opt/frontier/python/bin:${PATH}"
# runtime-python must be a wrapper, not a symlink (stage287 V3 interpreter fix)
RUN printf '%s\n' '#!/bin/sh' 'export PATH="/opt/frontier/python/bin:$PATH"' \
      'exec /opt/frontier/python/bin/python3 "$@"' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python
```

- Triton compiles a small C launcher at runtime against `Python.h`, so
  `python3-dev` and `gcc` are needed. The plain CUDA devel image does not
  guarantee a host compiler.
- If the Verifier also wants the strict installed-wheel validator, copy
  `python-wheel-lock.json` and `verify_python_lock.py` from the same stage287
  folder. This is optional, since `--require-hashes` already enforces the
  archive hashes.
- Nothing else needs a separate `pip install`. accelerate 1.12.0,
  datasets 4.4.1 and transformers 4.57.1 are already in the lock.

## Model staging (offline)

`../model-assets.json` pins `Qwen/Qwen3-4B` at revision
`1cfa9a7208912126459214e8b04321603b3df60c` (main at pin time, last modified
2025-07-26, Apache-2.0, not gated). It covers 10 inference files totalling
8,060,896,856 bytes. The three safetensors shards and `tokenizer.json` carry an
LFS sha256.

```dockerfile
RUN install -d -m 0755 /opt/frontier/models \
    && HF_HUB_OFFLINE=0 /opt/frontier/python/bin/python -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=["config.json","generation_config.json","merges.txt","model-00001-of-00003.safetensors","model-00002-of-00003.safetensors","model-00003-of-00003.safetensors","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","vocab.json"]; t="/opt/frontier/models/Qwen3-4B"; snapshot_download(repo_id="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c", local_dir=t, allow_patterns=files); [os.chmod(Path(t)/f, 0o444) for f in files]' \
    && find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`judge/config.yaml` snippet:

```yaml
  huggingface_models:
    - repo_id: Qwen/Qwen3-4B
      revision: 1cfa9a7208912126459214e8b04321603b3df60c
      target: Qwen3-4B
```

## Alternatives considered

- **(a) Official `vllm/vllm-openai:v0.11.0`.** Index digest
  `sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c`,
  amd64 manifest
  `sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a`.
  It was built on 2025-10-03 and has 12.45 GB of compressed layers. The image
  config confirms Python 3.12 (`PYTHON_VERSION=3.12`), CUDA 12.8.1 devel and
  Ubuntu 22.04, with FlashInfer v0.3.1 built in. torch 2.8.0 comes from
  vLLM's `requirements/cuda.txt` (`torch==2.8.0`). Its ENTRYPOINT is the
  OpenAI server, so we would have to reset it.
  - Rejected for two reasons. First, its transitive set was resolved unpinned
    at build time (vLLM only asks `transformers>=4.55.2`). The installed
    transformers, numpy and opencv versions cannot be read without pulling the
    layers. Second, any accelerate/datasets/transformers==4.57.1 overlay would
    then resolve against an inventory we do not know, so that overlay cannot be
    hash-locked. It is also about 2.4 times larger than the recommended base.
  - Use it as a fallback if the locked venv fails an import or kernel smoke
    test. In that case, add
    `pip install transformers==4.57.1 accelerate==1.12.0 datasets==4.4.1`,
    run `pip check`, and record `pip freeze` in the image.
- **(b) The vLLM task runtime** (`joyemang33/fcs2-task-vllm_llm_serving_optimization-runtime:h100-r001@sha256:2b09056416a3...`).
  It carries vLLM built from source at commit `b8b302cd` together with the
  Qwen3-Coder model and that task's runtime. The vLLM version is wrong for
  this task and the image is task-specific, so it is not reusable.
- **(c) `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`**
  (`sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6`,
  9.40 GB, Ubuntu 22.04).
  - Its Python is conda-based. The pytorch Dockerfile defaults to 3.11, which
    I did not verify from metadata, and it would not match the cp312 lock.
  - pip would also reinstall torch on top of conda's copy. It is larger than
    the recommended base for no gain.

## Risks

- apt packages are not version-pinned. That includes the python3 3.12.x patch
  level, which is the same situation as the other GPU tasks. The Python
  wheels are fully hash-pinned.
- The lock has never been installed. Remaining risks are ABI and kernel
  problems, for example xformers 0.0.32.post1 or vLLM's bundled FlashAttention
  on H100 with NumPy 2.2.6 and pandas 3.0.6. Gate the image on a GPU smoke
  test before calibration.
- There is no FlashInfer in the lock. vLLM 0.11.0 falls back to its bundled
  FlashAttention backend on H100, which is fine for Qwen3-4B, but the logs will
  warn about it.
- Image size is about 5.2 GB base, plus about 5 GB of wheels (compressed) and
  8.06 GB of model. Size the build disk and timeouts for that.

## Sources

- Registry API (anonymous token): `registry-1.docker.io/v2/{nvidia/cuda,vllm/vllm-openai,pytorch/pytorch}/manifests/<tag>` and the config blobs.
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/requirements/{cuda,common}.txt`; `https://pypi.org/pypi/vllm/0.11.0/json`.
- `https://raw.githubusercontent.com/adamzweiger/compaction/2020f6d37940fe013e7e7f1ebf409e5d107c5670/requirements.txt`.
- The authoring runtime-admission records (stage283 constraint receipts, stage287 review and lock).
- `https://huggingface.co/api/models/Qwen/Qwen3-4B` and `/tree/<sha>?recursive=true&expand=true`.

## As adopted in this task

- Both GPU Dockerfiles use this recipe verbatim.
- The lock lives at `agent-base/frontier/task/runtime/requirements-cp312-linux-x86_64.lock`.
  It is public, because the agent runner installs the same closure. It is
  stage287 except for `numpy==2.2.6` and `opencv-python-headless==4.12.0.88`
  (was 1.26.4 and 4.11.0.86), changed after the first H100 smoke.
- `LICENSE` is added to the Qwen3-4B `allow_patterns` so the Apache-2.0 notice
  ships with the weights.
