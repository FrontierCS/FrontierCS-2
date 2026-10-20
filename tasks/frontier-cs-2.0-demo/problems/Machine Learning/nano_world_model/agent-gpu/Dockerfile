# syntax=docker/dockerfile:1.7
# GPU runner image for nano_world_model. The platform creates one of these on the
# solver's request and connects it through its own SSH gateway; the solver never
# holds a provider credential and never learns this instance's address.
#
# It carries the toolchain and the read-only source tree, and no part of either
# suite: what runs here is the solver's own build and the task's benchmark, and a
# runner that could see the graded workload would be a second grading path.
# The publisher supplies an immutable image built from FrontierCS2's trusted
# `driver/images/gpu-ssh-base/Dockerfile` recipe.
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libnss-wrapper libseccomp2 openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir Pillow==10.4.0 PyYAML==6.0.2 accelerate==0.33.0 av==12.3.0 decord==0.6.0 diffusers==0.30.3 einops==0.8.0 h5py==3.11.0 huggingface-hub==0.34.4 hydra-core==1.3.2 imageio-ffmpeg==0.5.1 imageio==2.35.1 lpips==0.1.4 moviepy==1.0.3 numpy==1.26.4 omegaconf==2.3.0 opencv-python-headless==4.10.0.84 pandas==2.2.2 piqa==1.3.2 pytorch-fid==0.3.0 pytorch-lightning==1.9.5 requests==2.32.3 safetensors==0.4.5 scikit-image==0.24.0 scipy==1.13.1 tensorboard==2.17.1 timm==1.0.9 torch==2.4.1 torchmetrics==0.11.4 torchvision==0.19.1 tqdm==4.66.5 transformers==4.36.2 wandb==0.17.8 zstandard

RUN install -d -m 0755 /opt/frontier/models
COPY agent-base/frontier/task/runtime/model_asset_stager.py /opt/frontier/model_asset_stager.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'knightnemo/nanowm-b2-dino-wm-point-maze-30k'"'"', revision='"'"'b84303b7151cbd1836841c3a87bf095e1015b150'"'"', target='"'"'/opt/frontier/models/point-maze'"'"', files=[{'"'"'path'"'"': '"'"'.gitattributes'"'"', '"'"'bytes'"'"': 1519, '"'"'git_blob_id'"'"': '"'"'a6344aac8c09253b3b630fb776ae94478aa0275b'"'"'}, {'"'"'path'"'"': '"'"'README.md'"'"', '"'"'bytes'"'"': 1806, '"'"'git_blob_id'"'"': '"'"'1297ba4ae78365acfb5bd087f2ebe7637a2ca681'"'"'}, {'"'"'path'"'"': '"'"'config.yaml'"'"', '"'"'bytes'"'"': 2784, '"'"'git_blob_id'"'"': '"'"'446b9fa8e5c98f7f30d4a01f8986cf19744db822'"'"'}, {'"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'bytes'"'"': 634416296, '"'"'sha256'"'"': '"'"'86ac17ff150da027618476817ca48708014de6f5d8d09277fe352e2584f54f28'"'"', '"'"'git_blob_id'"'"': '"'"'86658493fbc58539b9b85331df8d295e40119295'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'stabilityai/sd-vae-ft-mse'"'"', revision='"'"'31f26fdeee1355a5c34592e401dd41e45d25a493'"'"', target='"'"'/opt/frontier/models/sd-vae'"'"', files=[{'"'"'path'"'"': '"'"'.gitattributes'"'"', '"'"'bytes'"'"': 1456, '"'"'git_blob_id'"'"': '"'"'99f243ff7947dfc5afddaf3d818da3d0124d5ca2'"'"'}, {'"'"'path'"'"': '"'"'README.md'"'"', '"'"'bytes'"'"': 6844, '"'"'git_blob_id'"'"': '"'"'61d3c7cf571c574a406544612b857ac1b26f4439'"'"'}, {'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 547, '"'"'git_blob_id'"'"': '"'"'0db26717579be63eb0ddbf15b43faa43700dfe5a'"'"'}, {'"'"'path'"'"': '"'"'diffusion_pytorch_model.bin'"'"', '"'"'bytes'"'"': 334707217, '"'"'sha256'"'"': '"'"'1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc'"'"', '"'"'git_blob_id'"'"': '"'"'ba36f34d64ad3be997b7cab94b0b9acd61272851'"'"'}, {'"'"'path'"'"': '"'"'diffusion_pytorch_model.safetensors'"'"', '"'"'bytes'"'"': 334643276, '"'"'sha256'"'"': '"'"'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815'"'"', '"'"'git_blob_id'"'"': '"'"'90464d67ac7303d0ee4696334df13da130a948ea'"'"'}], previous_files=[])'
RUN find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN install -d -m 0555 /opt/frontier/models/torch/hub/checkpoints \
    && python3 -c 'import torch; torch.hub.load_state_dict_from_url('"'"'https://download.pytorch.org/models/vgg16-397923af.pth'"'"', model_dir='"'"'/opt/frontier/models/torch/hub/checkpoints'"'"', file_name='"'"'vgg16-397923af.pth'"'"', check_hash=True, progress=False)' \
    && chmod -R a+rX,a-w /opt/frontier/models/torch
ENV TORCH_HOME=/opt/frontier/models/torch

RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner nano_world_model" \
      dev.frontier.problem-id="nano_world_model" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime

# Snapshot only public toolchain/model settings, never the build environment's
# credentials. SSH has a deliberately minimal environment.
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path('"'"'/opt/frontier/runtime-python'"'"'); output = path.open('"'"'x'"'"'); output.write('"'"'#!/bin/sh\nexec '"'"' + shlex.quote(sys.executable) + '"'"' "$@"\n'"'"'); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} +

COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
