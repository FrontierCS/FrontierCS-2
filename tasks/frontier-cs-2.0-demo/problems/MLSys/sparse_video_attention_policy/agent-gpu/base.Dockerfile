# syntax=docker/dockerfile:1.7
# GPU runner image for sparse_video_attention_policy. The platform creates one of these on the
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
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates cmake ffmpeg g++-11 gcc-11 git libnss-wrapper make openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir accelerate==1.10.1 cmake==4.1.0 cuvs-cu12==26.2.0 diffusers==0.34.0 flashinfer-python==0.2.10 ftfy==6.3.1 huggingface-hub==0.34.4 imageio-ffmpeg==0.6.0 imageio==2.37.0 loguru==0.7.3 lpips==0.1.4 numpy==2.1.2 pillow==11.0.0 protobuf==6.32.0 setuptools==80.9.0 sympy==1.13.1 termcolor==3.1.0 torch==2.6.0 torchvision==0.21.0 transformers==4.57.6 triton==3.2.0 zstandard

RUN install -d -m 0755 /opt/frontier/models
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'.gitattributes'"'"', '"'"'README.md'"'"', '"'"'assets/comp_effic.png'"'"', '"'"'assets/data_for_diff_stage.jpg'"'"', '"'"'assets/i2v_res.png'"'"', '"'"'assets/logo.png'"'"', '"'"'assets/t2v_res.jpg'"'"', '"'"'assets/vben_1.3b_vs_sota.png'"'"', '"'"'assets/vben_vs_sota.png'"'"', '"'"'assets/video_dit_arch.jpg'"'"', '"'"'assets/video_vae_res.jpg'"'"', '"'"'examples/i2v_input.JPG'"'"', '"'"'model_index.json'"'"', '"'"'scheduler/scheduler_config.json'"'"', '"'"'text_encoder/config.json'"'"', '"'"'text_encoder/model-00001-of-00005.safetensors'"'"', '"'"'text_encoder/model-00002-of-00005.safetensors'"'"', '"'"'text_encoder/model-00003-of-00005.safetensors'"'"', '"'"'text_encoder/model-00004-of-00005.safetensors'"'"', '"'"'text_encoder/model-00005-of-00005.safetensors'"'"', '"'"'text_encoder/model.safetensors.index.json'"'"', '"'"'tokenizer/special_tokens_map.json'"'"', '"'"'tokenizer/spiece.model'"'"', '"'"'tokenizer/tokenizer.json'"'"', '"'"'tokenizer/tokenizer_config.json'"'"', '"'"'transformer/config.json'"'"', '"'"'transformer/diffusion_pytorch_model-00001-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00002-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00003-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00004-of-00012.safetensors'"'"']; target='"'"'/opt/frontier/models/Wan2.1-T2V-14B-Diffusers'"'"'; snapshot_download(repo_id='"'"'Wan-AI/Wan2.1-T2V-14B-Diffusers'"'"', revision='"'"'38ec498cb3208fb688890f8cc7e94ede2cbd7f68'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'transformer/diffusion_pytorch_model-00005-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00006-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00007-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00008-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00009-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00010-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00011-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00012-of-00012.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model.safetensors.index.json'"'"', '"'"'vae/config.json'"'"', '"'"'vae/diffusion_pytorch_model.safetensors'"'"']; target='"'"'/opt/frontier/models/Wan2.1-T2V-14B-Diffusers'"'"'; snapshot_download(repo_id='"'"'Wan-AI/Wan2.1-T2V-14B-Diffusers'"'"', revision='"'"'38ec498cb3208fb688890f8cc7e94ede2cbd7f68'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
# HunyuanVideo diffusers subset (Tencent Hunyuan Community License; see
# docs/task-authoring/sparse_video_attention_policy/hunyuan/LICENSE), pinned by
# revision. model-assets.json lists every file with its size and LFS sha256.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'LICENSE'"'"', '"'"'Notice'"'"', '"'"'README.md'"'"', '"'"'model_index.json'"'"', '"'"'scheduler/scheduler_config.json'"'"', '"'"'text_encoder/config.json'"'"', '"'"'text_encoder/model-00001-of-00004.safetensors'"'"', '"'"'text_encoder/model-00002-of-00004.safetensors'"'"', '"'"'text_encoder/model-00003-of-00004.safetensors'"'"', '"'"'text_encoder/model-00004-of-00004.safetensors'"'"', '"'"'text_encoder/model.safetensors.index.json'"'"', '"'"'text_encoder_2/config.json'"'"', '"'"'text_encoder_2/model.safetensors'"'"', '"'"'tokenizer/special_tokens_map.json'"'"', '"'"'tokenizer/tokenizer.json'"'"', '"'"'tokenizer/tokenizer_config.json'"'"', '"'"'tokenizer_2/merges.txt'"'"', '"'"'tokenizer_2/special_tokens_map.json'"'"', '"'"'tokenizer_2/tokenizer_config.json'"'"', '"'"'tokenizer_2/vocab.json'"'"', '"'"'vae/config.json'"'"', '"'"'vae/diffusion_pytorch_model.safetensors'"'"']; target='"'"'/opt/frontier/models/HunyuanVideo'"'"'; snapshot_download(repo_id='"'"'tencent/HunyuanVideo'"'"', revision='"'"'2a15b5574ee77888e51ae6f593b2ceed8ce813e5'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'transformer/config.json'"'"', '"'"'transformer/diffusion_pytorch_model-00001-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00002-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00003-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00004-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00005-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00006-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model.safetensors.index.json'"'"']; target='"'"'/opt/frontier/models/HunyuanVideo'"'"'; snapshot_download(repo_id='"'"'tencent/HunyuanVideo'"'"', revision='"'"'2a15b5574ee77888e51ae6f593b2ceed8ce813e5'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
RUN find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN install -d -m 0555 /opt/frontier/models/torch/hub/checkpoints \
    && python3 -c 'import torch; torch.hub.load_state_dict_from_url('"'"'https://download.pytorch.org/models/vgg16-397923af.pth'"'"', model_dir='"'"'/opt/frontier/models/torch/hub/checkpoints'"'"', file_name='"'"'vgg16-397923af.pth'"'"', check_hash=True, progress=False)' \
    && chmod -R a+rX,a-w /opt/frontier/models/torch
ENV TORCH_HOME=/opt/frontier/models/torch

RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner sparse_video_attention_policy" \
      dev.frontier.problem-id="sparse_video_attention_policy" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} +

COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

# Keep the build interpreter discoverable after SSH intentionally clears PATH.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import torch, cmake'

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
