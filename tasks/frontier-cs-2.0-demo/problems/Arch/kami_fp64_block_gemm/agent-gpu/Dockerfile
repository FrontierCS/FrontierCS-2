# syntax=docker/dockerfile:1.7
# GPU runner image for kami_fp64_block_gemm. The platform creates one of these on the
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
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir zstandard


RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner kami_fp64_block_gemm" \
      dev.frontier.problem-id="kami_fp64_block_gemm" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-base/frontier/task/repo/README.md /frontier/task/repo/README.md

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} +

COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
