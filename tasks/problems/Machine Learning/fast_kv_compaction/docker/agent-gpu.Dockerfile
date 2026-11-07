# syntax=docker/dockerfile:1.7
# Agent GPU runner for fast_kv_compaction: the offline Qwen3-4B runtime and the
# read-only public tree (12 public articles), nothing from the judge. The
# platform connects it through its own SSH gateway; the solver never holds a
# provider credential. Same runtime base and Python closure (docker/gpu-requirements.lock,
# bind-mounted for the build) as docker/verifier-gpu.Dockerfile; see
# docs/task-authoring/fast_kv_compaction/agent-gpu-BASE_IMAGE.md. The SSH bootstrap comes from
# the shared gpu-tools image. Build context: this task's root, with .frontier-generated/.
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
# nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 manifest
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

USER root
ARG DEBIAN_FRONTEND=noninteractive
# Python 3.12 venv from the hashed 149-wheel closure (torch 2.8.0, vLLM 0.11.0,
# transformers 4.57.1, accelerate 1.12.0, datasets 4.4.1); see BASE_IMAGE.md.
RUN --mount=type=bind,source=docker/gpu-requirements.lock,target=/tmp/frontier-python-lock/requirements.lock \
    apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates gcc g++ git libnss-wrapper \
       openssh-client openssh-server python3 python3-dev python3-pip python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --only-binary=:all: --require-hashes --ignore-installed \
       --index-url https://pypi.org/simple -r /tmp/frontier-python-lock/requirements.lock \
    && /opt/frontier/python/bin/python -m pip check
ENV PATH="/opt/frontier/python/bin:${PATH}"
# runtime-python is a wrapper, not a symlink, so the venv stays active over SSH.
RUN printf '%s\n' '#!/bin/sh' 'export PATH="/opt/frontier/python/bin:$PATH"' \
      'exec /opt/frontier/python/bin/python3 "$@"' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python

# Qwen/Qwen3-4B (Apache-2.0, LICENSE shipped), pinned by revision;
# model-assets.json lists every file with its size and LFS sha256.
RUN install -d -m 0755 /opt/frontier/models \
    && HF_HUB_OFFLINE=0 /opt/frontier/python/bin/python -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=["config.json","generation_config.json","LICENSE","merges.txt","model-00001-of-00003.safetensors","model-00002-of-00003.safetensors","model-00003-of-00003.safetensors","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","vocab.json"]; t="/opt/frontier/models/Qwen3-4B"; snapshot_download(repo_id="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c", local_dir=t, allow_patterns=files); [os.chmod(Path(t)/f, 0o444) for f in files]' \
    && rm -rf /opt/frontier/models/Qwen3-4B/.cache \
    && find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

LABEL org.opencontainers.image.title="FrontierCS2 GPU runner fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction"

COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

COPY repo /frontier/task/repo
COPY gpu/task_runtime.py /opt/frontier/task-runtime
RUN /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} +

ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 10001:10001
WORKDIR /workspace
