# syntax=docker/dockerfile:1.7
# Agent GPU runner image of kami_fp64_block_gemm. The platform creates one on the solver's
# request and connects it through its own SSH gateway; the solver never holds a provider
# credential. It carries the CUDA 12.8 toolchain and the read-only source tree, and no part of
# either suite. The SSH bootstrap comes from the shared gpu-tools image; zstandard is pinned to
# the previous release's version. /opt/frontier/task-runtime (gpu/task_runtime.py, new with the
# #757 migration) runs task commands with the image's toolchain environment, as on the other
# GPU tasks. Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir zstandard==0.25.0

RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

LABEL org.opencontainers.image.title="FrontierCS2 GPU runner kami_fp64_block_gemm" \
      dev.frontier.problem-id="kami_fp64_block_gemm" \
      dev.frontier.generator="frontiersmith2"

COPY repo /frontier/task/repo

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} +

COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

COPY gpu/task_runtime.py /opt/frontier/task-runtime
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
