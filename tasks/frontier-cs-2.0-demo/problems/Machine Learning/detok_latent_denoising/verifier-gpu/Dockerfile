# syntax=docker/dockerfile:1.7
# Verifier for detok_latent_denoising: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
# Reproduce the receipt-bound dependency base from its original public CUDA
# image and exact dependency/assets inputs; see docs/acceptance for provenance.
FROM nvidia/cuda@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates git tar openssh-client \
    && rm -rf /var/lib/apt/lists/*
COPY agent-base/frontier/task/runtime/dependency-base/dependencies /opt/frontier/dependencies/
COPY agent-base/frontier/task/runtime/dependency-base/runtime-layer.lock /opt/frontier/dependencies/runtime-layer.lock
RUN python3 -m pip install --no-cache-dir --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r /opt/frontier/dependencies/runtime-layer.lock

COPY agent-base/frontier/task/runtime/dependency-base/assets /opt/frontier/models/
COPY agent-base/frontier/task/runtime/dependency-base/SHA256SUMS /opt/frontier/dependencies/SHA256SUMS
RUN cd / && sha256sum -c /opt/frontier/dependencies/SHA256SUMS \
    && chmod -R a+rX,a-w /opt/frontier/models /opt/frontier/dependencies
ENV TORCH_HOME=/opt/frontier/models/torch \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY agent-base/frontier/task/runtime/dependency-base/record_image_packages.py /opt/frontier/build-evidence/record_image_packages.py
RUN useradd --uid 10001 --no-create-home --home-dir /workspace --shell /usr/sbin/nologin frontier \
    && python3 /opt/frontier/build-evidence/record_image_packages.py \
    && chmod -R a+rX,a-w /opt/frontier/build-evidence

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir annotated-types==0.8.0 anyio==4.14.2 certifi==2026.7.22 charset-normalizer==3.5.1 click==8.5.0 diffusers==0.35.1 einops==0.8.1 exceptiongroup==1.3.1 filelock==3.32.6 fsspec==2026.7.0 gitdb==4.0.12 gitpython==3.1.62 h11==0.16.0 hf-xet==1.6.0 httpcore==1.0.9 httpx==0.28.1 huggingface-hub==1.30.0 idna==3.19 importlib-metadata==9.0.1 jinja2==3.1.6 markdown-it-py==4.2.0 markupsafe==3.0.3 mdurl==0.1.2 mpmath==1.3.0 networkx==3.4.2 numpy==1.26.4 nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 nvidia-cusparselt-cu12==0.6.2 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127 nvidia-nvtx-cu12==12.4.127 packaging==26.3 pillow==11.3.0 platformdirs==4.11.8 protobuf==6.33.6 pydantic-core==2.46.5 pydantic==2.13.5 pygments==2.21.0 pyyaml==6.0.3 regex==2026.9.3 requests==2.32.5 rich==14.1.0 safetensors==0.8.0 scipy==1.15.3 sentry-sdk==2.69.1 smmap==5.0.3 sympy==1.13.1 'torch-fidelity @ file:///opt/frontier/dependencies/torch_fidelity-0.4.0b0-py3-none-any.whl#sha256=00622864889f5fd6eb3ae5bc67fd9cd93c89f9590a672fa769abbfca3e87908c' torch==2.6.0+cu124 torchdiffeq==0.2.5 torchvision==0.21.0+cu124 tqdm==4.67.1 triton==3.2.0 typing-extensions==4.15.0 typing-inspection==0.4.4 urllib3==2.7.0 wandb==0.21.1 zipp==4.1.0 zstandard zstandard==0.25.0

RUN install -d -m 0755 /opt/frontier/models
COPY agent-base/frontier/task/runtime/model_asset_stager.py /opt/frontier/model_asset_stager.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'jjiaweiyang/l-DeTok'"'"', revision='"'"'8e6475697ac1505fe39d35221836f55e4fbc6ce5'"'"', target='"'"'/opt/frontier/models/detok'"'"', files=[{'"'"'path'"'"': '"'"'detok-BB-gamm3.0-m0.7.pth'"'"', '"'"'bytes'"'"': 3040937682, '"'"'sha256'"'"': '"'"'8e0790fd04a2f70db764db0ecea56ad844a2587062eded1ce8c7b34a654638df'"'"', '"'"'git_blob_id'"'"': '"'"'e65de32d9ceaf5d187a0f63d8a82081fd6545ed8'"'"'}], previous_files=[])'
RUN find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN install -d -m 0555 /opt/frontier/models/torch/hub/checkpoints \
    && python3 -c 'import torch; torch.hub.load_state_dict_from_url('"'"'https://download.pytorch.org/models/vgg16-397923af.pth'"'"', model_dir='"'"'/opt/frontier/models/torch/hub/checkpoints'"'"', file_name='"'"'vgg16-397923af.pth'"'"', check_hash=True, progress=False)' \
    && python3 -c 'import torch; torch.hub.load_state_dict_from_url('"'"'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth'"'"', model_dir='"'"'/opt/frontier/models/torch/hub/checkpoints'"'"', file_name='"'"'weights-inception-2015-12-05-6726825d.pth'"'"', check_hash=True, progress=False)' \
    && chmod -R a+rX,a-w /opt/frontier/models/torch
ENV TORCH_HOME=/opt/frontier/models/torch

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier detok_latent_denoising" \
      dev.frontier.problem-id="detok_latent_denoising" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py

# Source snapshots are created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user, so merely removing write bits turns 0600 files into 0400 root-only
# files. Grant read/traverse while preserving executable files; the bundle
# stays immutable and becomes usable by both platform UIDs.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap


USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=L4 FRONTIER_VERIFIER_GPU_SECONDS=5400 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
