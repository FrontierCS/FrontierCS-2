# syntax=docker/dockerfile:1.7
# Verifier GPU runner image of fast_kv_compaction: the offline Qwen3-4B runtime, the isolated
# compaction worker and the judge. The trusted command test sandbox [test_sandboxes.shard]: the
# control image (docker/verifier.Dockerfile) leases one single-H100 runner per shard from this
# image and runs `evaluator.py --shard-run` there on the applied tree; task code never receives
# a provider credential. Runtime base and Python closure:
# docs/task-authoring/fast_kv_compaction/verifier-gpu-BASE_IMAGE.md.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
# nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 manifest
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

USER root
ARG DEBIAN_FRONTEND=noninteractive
# Python 3.12 venv from the hashed 149-wheel closure (torch 2.8.0, vLLM 0.11.0,
# transformers 4.57.1, accelerate 1.12.0, datasets 4.4.1); see BASE_IMAGE.md.
COPY docker/gpu-requirements.lock /opt/frontier/python-lock/requirements.lock
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates gcc g++ git libnss-wrapper \
       openssh-client openssh-server iproute2 python3 python3-dev python3-pip python3-venv sudo tar util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --only-binary=:all: --require-hashes --ignore-installed \
       --index-url https://pypi.org/simple -r /opt/frontier/python-lock/requirements.lock \
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

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

LABEL org.opencontainers.image.title="FrontierCS2 verifier fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction"

# The submitted method runs as the harness's own user, but only inside the
# namespaces built by the launcher (no root, sudo or cgroup needed): fresh user,
# mount, network, PID and IPC namespaces; tmpfs over the judge, the verifier
# input/output and every scratch path; then a nested user namespace with no
# capabilities, so those mounts are locked. It can read the model and the
# worker code below, but not the judge. WORKER_ROOT is its read-only tree mount.
COPY gpu/fastkv-worker /opt/frontier/fastkv-worker
COPY judge/compaction_worker.py /opt/frontier/worker/compaction_worker.py
COPY judge/wire.py /opt/frontier/worker/wire.py
COPY judge/gpu_probe.py /opt/frontier/worker/gpu_probe.py
COPY judge/isolation_probe.py /opt/frontier/worker/isolation_probe.py
RUN chown root:root /opt/frontier/fastkv-worker \
    && chmod 0555 /opt/frontier/fastkv-worker \
    && install -d -m 0755 /opt/frontier/worker-root \
    && chmod -R a+rX,a-w /opt/frontier/worker

# The judge (also the mount point the worker's namespace hides) and the starter; each shard runs
# the uploaded copy of the judge on the uploaded applied tree.
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json

RUN /opt/frontier/runtime-python -c 'import zstandard, torch, vllm, transformers' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py \
    && chmod 0555 /opt/frontier/verifier \
    && chown -R root:frontier /opt/frontier/task \
    && chmod -R u=rwX,g=rX,o= /opt/frontier/task \
    && chmod a-w -R /opt/frontier/task
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
