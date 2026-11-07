# syntax=docker/dockerfile:1.7
# Agent GPU runner image of ivf_pq_gpu_kernel_optimization. The platform creates one on the
# solver's request and connects it through its own SSH gateway; it carries the toolchain and
# the read-only source tree, and no part of either suite. The SSH bootstrap comes from the
# shared gpu-tools image (tasks/shared/bases.json); h5py is pinned as before. Build context:
# this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
# The frozen SIFT-128 bytes (an export of the original Modal volume) come from this task's
# previous published Agent GPU image (copied by digest), and are checked by sha256 below.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client openssh-server tar \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace
RUN python3 -m pip install --no-cache-dir h5py==3.14.0
COPY --from=docker.io/joyemang33/fcs2-task-ivf_pq_gpu_kernel_optimization-agent-gpu@sha256:903057eca3af1bd577ea6232a9f9338ad554b9a7e3a8a54700e8bfea6360f4a9 /data/sift.hdf5 /data/sift.hdf5
RUN printf '%s\n' 'dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984  /data/sift.hdf5' | sha256sum -c - \
    && chmod 0444 /data/sift.hdf5
LABEL org.opencontainers.image.title="FrontierCS2 IVF-PQ Agent H100 runtime" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="ivf_pq_gpu_kernel_optimization"
COPY repo /frontier/task/repo
COPY repo/benchmark /opt/frontier/development
COPY gpu/task_runtime.py /opt/frontier/task-runtime
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} +
# The trusted bootstrap starts as root, then OpenSSH runs task commands as frontier.
USER root
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
