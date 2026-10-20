# syntax=docker/dockerfile:1.7
# Verifier for r_top01_2604-28190-cand_000-preflight-8250ab0a4a09: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

USER root
ARG DEBIAN_FRONTEND=noninteractive
COPY verifier-gpu/python-constraints.txt /opt/frontier/python-constraints.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir -c /opt/frontier/python-constraints.txt exceptiongroup==1.3.1 diffusers einops huggingface-hub==0.34.4 matplotlib muon-optimizer numpy opencv-python-headless pandas pillow pyyaml requests rich scipy tabulate timm torch==2.6.0 torchvision==0.21.0 tqdm typing_extensions wandb zstandard

RUN install -d -m 0755 /opt/frontier/models
COPY agent-base/frontier/task/runtime/model_asset_stager.py /opt/frontier/model_asset_stager.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'jjiaweiyang/FD-Loss'"'"', revision='"'"'88ce2bb4061b0dd58839a1827235ffcffa51eed2'"'"', target='"'"'/opt/frontier/models/fd-loss'"'"', files=[{'"'"'path'"'"': '"'"'checkpoints/base/pMF-B_256.pth'"'"', '"'"'bytes'"'"': 473012794, '"'"'sha256'"'"': '"'"'aa95e128b3378e25285f9c2470d1e074d73a8d9d7846a7c9ab59e851c2ed64f8'"'"', '"'"'git_blob_id'"'"': '"'"'6a3f54f376e73dadb10d3f9009d8c9ca68c0cf16'"'"'}, {'"'"'path'"'"': '"'"'data/fid_stats/paper_ref_stats.pkl'"'"', '"'"'bytes'"'"': 147164490, '"'"'sha256'"'"': '"'"'fe994d2be02d97b7494c393e0e4b283142746e18be48a5a34fc0a8f415748da3'"'"', '"'"'git_blob_id'"'"': '"'"'3d8e4710aa0a213b563ef458f0beb194a6aebad2'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'timm/vit_large_patch16_224.mae'"'"', revision='"'"'126727e9b8ccd632d08a0d75fff095017c225e54'"'"', target='"'"'/opt/frontier/models/mae'"'"', files=[{'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 609, '"'"'git_blob_id'"'"': '"'"'d4444d919200b99125a280c723986e2cceaea0e0'"'"'}, {'"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'bytes'"'"': 1213234508, '"'"'sha256'"'"': '"'"'d58a1a547a07c3c41917c23069dcaadab6f469a8cf793a3cba5fe404c344ab68'"'"', '"'"'git_blob_id'"'"': '"'"'a5152d64b138714fc2fa7e7bee799c6ce2dc3a1c'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'timm/vit_so400m_patch16_siglip_256.v2_webli'"'"', revision='"'"'a8dedb158968170d6fb45a000e0d5755d8021386'"'"', target='"'"'/opt/frontier/models/siglip2'"'"', files=[{'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 577, '"'"'git_blob_id'"'"': '"'"'0821c874d0eaca12da00c2c6375cbd1f4ae6d674'"'"'}, {'"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'bytes'"'"': 1711584976, '"'"'sha256'"'"': '"'"'68c976f5691fd6a20187da6b3c6132601dba52242c349acf8534c64b99ab2b96'"'"', '"'"'git_blob_id'"'"': '"'"'f2ec3cb9371c227206eb79d9890350de5d9df1fc'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'timm/convnextv2_base.fcmae_ft_in22k_in1k'"'"', revision='"'"'9e250ed8f88b436472d2b24af26f82a8aa8c719d'"'"', target='"'"'/opt/frontier/models/convnextv2'"'"', files=[{'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 902, '"'"'git_blob_id'"'"': '"'"'51c4f7e6971a6e3b9cbcd520af4a8d536b5a99dc'"'"'}, {'"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'bytes'"'"': 354909632, '"'"'sha256'"'"': '"'"'3b04d2fd7f5268b706210542bd6b6833ad75d6a5e4612a1ae34e8414dbd50b81'"'"', '"'"'git_blob_id'"'"': '"'"'e28808a1ba2455248db8e3169b10ddcaf502d9ad'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'timm/vit_large_patch14_dinov2.lvd142m'"'"', revision='"'"'4741e1cafbf45415e77074bb0cb42dba76c8684a'"'"', target='"'"'/opt/frontier/models/dinov2'"'"', files=[{'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 616, '"'"'git_blob_id'"'"': '"'"'5873223cd12446cc5371ecb58a6df8b028df877a'"'"'}, {'"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'bytes'"'"': 1217502758, '"'"'sha256'"'"': '"'"'0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355'"'"', '"'"'git_blob_id'"'"': '"'"'2160941f9c44397ab51a7c5e7ccd88fce825cead'"'"'}], previous_files=[])'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'timm/vit_large_patch14_clip_224.openai'"'"', revision='"'"'18d0535469bb561bf468d76c1d73aa35156c922b'"'"', target='"'"'/opt/frontier/models/clip'"'"', files=[{'"'"'path'"'"': '"'"'config.json'"'"', '"'"'bytes'"'"': 619, '"'"'git_blob_id'"'"': '"'"'b3fe5e4ed648cb0a2be17b4ce0a994b8bbea95c0'"'"'}, {'"'"'path'"'"': '"'"'pytorch_model.bin'"'"', '"'"'bytes'"'"': 1710633285, '"'"'sha256'"'"': '"'"'eb70c15b117effa76315db43fcbd9f32a655292fc861798343939a02f561c6ae'"'"', '"'"'git_blob_id'"'"': '"'"'71d45c096bebc157d7fb813558b31fc554c41dee'"'"'}], previous_files=[])'
RUN find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN install -d -m 0555 /opt/frontier/models/torch/hub/checkpoints \
    && python3 -c 'import torch; torch.hub.load_state_dict_from_url('"'"'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth'"'"', model_dir='"'"'/opt/frontier/models/torch/hub/checkpoints'"'"', file_name='"'"'weights-inception-2015-12-05-6726825d.pth'"'"', check_hash=True, progress=False)' \
    && chmod -R a+rX,a-w /opt/frontier/models/torch
ENV TORCH_HOME=/opt/frontier/models/torch

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier r_top01_2604-28190-cand_000-preflight-8250ab0a4a09" \
      dev.frontier.problem-id="r_top01_2604-28190-cand_000-preflight-8250ab0a4a09" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY agent-base/frontier/task/repo/README.md /opt/frontier/task/harbor/app/README.md
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
    && chmod 0555 /opt/frontier \
    && chmod 0444 /opt/frontier/*.py \
    && chmod -R a+rX,a-w /opt/frontier/task
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap


USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_SECONDS=7200 FRONTIER_VERIFIER_GPU_EVAL_SECONDS=10800 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
