# syntax=docker/dockerfile:1.7
# GPU Verifier image of sparse_video_attention_policy: the pinned models and toolchain, the
# judge, and the shared verifier modules copied from the digest-pinned verifier base (v9 in
# tasks/shared/bases.json). The SSH bootstrap comes from the shared gpu-tools image; pip
# versions are held to the previous release's by docker/verifier-gpu.constraints.txt.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,source=docker/verifier-gpu.constraints.txt,target=/tmp/frontier-pip-constraints.txt \
    apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates cmake ffmpeg g++-11 gcc-11 git libnss-wrapper make openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir --constraint /tmp/frontier-pip-constraints.txt accelerate==1.10.1 cmake==4.1.0 cuvs-cu12==26.2.0 diffusers==0.34.0 flashinfer-python==0.2.10 ftfy==6.3.1 huggingface-hub==0.34.4 imageio-ffmpeg==0.6.0 imageio==2.37.0 loguru==0.7.3 lpips==0.1.4 numpy==2.1.2 pillow==11.0.0 protobuf==6.32.0 setuptools==80.9.0 sympy==1.13.1 termcolor==3.1.0 torch==2.6.0 torchvision==0.21.0 transformers==4.57.6 triton==3.2.0 zstandard apache-tvm-ffi==0.1.13.post3 cuda-bindings==12.9.7 cuda-pathfinder==1.8.1 nvidia-cuda-nvdisasm==13.4.49 nvidia-cudnn-frontend==1.29.0 nvidia-cutlass-dsl==4.7.1 nvidia-cutlass-dsl-libs-base==4.7.1 nvidia-cutlass-dsl-libs-core==4.7.1 nvidia-cutlass-dsl-libs-cu12==4.7.1 nvidia-cutlass-dsl-libs-cu13==4.7.1 nvidia-ml-py==13.610.43 nvidia-nvshmem-cu12==3.7.2 regex==2026.9.10 typing-extensions==4.12.2

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

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier sparse_video_attention_policy" \
      dev.frontier.problem-id="sparse_video_attention_policy" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. pristine/ is the same repo/ tree the Agent image publishes;
# the shared verifier overlays the submission's allowed files on a copy of it.
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
# The shared verifier (edit_scope, then check_tree, then evaluate(tree)), from the pinned base.
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml

# Source snapshots are created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user, so merely removing write bits turns 0600 files into 0400 root-only
# files. Grant read/traverse while preserving executable files; the bundle
# stays immutable and becomes usable by both platform UIDs.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
# The judge runs every generation of submitted code with the shared isolate launcher
# (userns backend on the runner): its own namespaces, no judge files or targets, one GPU, the
# offline models read-only.
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
# /opt/frontier on the judge interpreter's default import path (frontier.pth), so the judge and
# every worker process it starts import isolate by name.
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0444 /opt/frontier/isolate.py

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=4
WORKDIR /frontier
