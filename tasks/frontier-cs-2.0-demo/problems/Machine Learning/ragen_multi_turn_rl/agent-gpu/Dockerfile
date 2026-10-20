# syntax=docker/dockerfile:1.7
# GPU runner image for ragen_multi_turn_rl. The platform creates one of these on the
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
FROM hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2@sha256:41dc66525bd551be71cb8abe9c1dd8e93c6bc1620c335038bc256ac4cbc09beb

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir anthropic==0.49.0 codetiming==1.4.0 gym-sokoban==0.0.6 gym==0.26.2 'gymnasium[toy-text]==1.1.1' huggingface-hub==0.34.4 hydra-core==1.3.2 numpy==1.26.4 pillow==11.1.0 setuptools==80.9.0 tensordict==0.6.2 together==1.5.8 torch==2.6.0 torchdata==0.11.0 tqdm==4.67.1 transformers==4.51.3 wheel==0.45.1 zstandard

RUN install -d -m 0755 /opt/frontier/models
COPY agent-base/frontier/task/bin/model_asset_stager.py /opt/frontier/model_asset_stager.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import sys; sys.path.insert(0, '"'"'/opt/frontier'"'"'); from model_asset_stager import stage_files; stage_files(repo_id='"'"'Qwen/Qwen2.5-0.5B-Instruct'"'"', revision='"'"'7ae557604adf67be50417f59c2c2f167def9a775'"'"', target='"'"'/opt/frontier/models/qwen2.5-0.5b-instruct'"'"', files=[{'"'"'bytes'"'"': 1519, '"'"'git_blob_id'"'"': '"'"'a6344aac8c09253b3b630fb776ae94478aa0275b'"'"', '"'"'path'"'"': '"'"'.gitattributes'"'"'}, {'"'"'bytes'"'"': 11343, '"'"'git_blob_id'"'"': '"'"'6634c8cc3133b3848ec74b9f275acaaa1ea618ab'"'"', '"'"'path'"'"': '"'"'LICENSE'"'"'}, {'"'"'bytes'"'"': 4917, '"'"'git_blob_id'"'"': '"'"'4b8373851d093eb9f3017443f27781c6971eff24'"'"', '"'"'path'"'"': '"'"'README.md'"'"'}, {'"'"'bytes'"'"': 659, '"'"'git_blob_id'"'"': '"'"'0dbb161213629a23f0fc00ef286e6b1e366d180f'"'"', '"'"'path'"'"': '"'"'config.json'"'"'}, {'"'"'bytes'"'"': 242, '"'"'git_blob_id'"'"': '"'"'dfc11073787daf1b0f9c0f1499487ab5f4c93738'"'"', '"'"'path'"'"': '"'"'generation_config.json'"'"'}, {'"'"'bytes'"'"': 1671839, '"'"'git_blob_id'"'"': '"'"'20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'"'"', '"'"'path'"'"': '"'"'merges.txt'"'"'}, {'"'"'bytes'"'"': 988097824, '"'"'git_blob_id'"'"': '"'"'d7db405a3f0d9bf1ba5bdd4e4211db8022ebe4eb'"'"', '"'"'path'"'"': '"'"'model.safetensors'"'"', '"'"'sha256'"'"': '"'"'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'"'"'}, {'"'"'bytes'"'"': 7031645, '"'"'git_blob_id'"'"': '"'"'443909a61d429dff23010e5bddd28ff530edda00'"'"', '"'"'path'"'"': '"'"'tokenizer.json'"'"'}, {'"'"'bytes'"'"': 7305, '"'"'git_blob_id'"'"': '"'"'07bfe0640cb5a0037f9322287fbfc682806cf672'"'"', '"'"'path'"'"': '"'"'tokenizer_config.json'"'"'}, {'"'"'bytes'"'"': 2776833, '"'"'git_blob_id'"'"': '"'"'4783fe10ac3adce15ac8f358ef5462739852c569'"'"', '"'"'path'"'"': '"'"'vocab.json'"'"'}], previous_files=[])'
RUN find /opt/frontier/models -type d -exec chmod 0555 {} +
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner ragen_multi_turn_rl" \
      dev.frontier.problem-id="ragen_multi_turn_rl" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime

# Snapshot only public toolchain/model settings, never the build environment's
# credentials. SSH has a deliberately minimal environment.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
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
