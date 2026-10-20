# syntax=docker/dockerfile:1.7
# GPU runner image for flow_opd. The platform creates one of these on the
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
FROM docker.io/joyemang33/fcs2-task-flow_opd-gpu-runtime@sha256:47687f9d3935b55d321303ada053de92f435c70ac5f497952fcc8e3e6f914da5 AS flow_opd_runtime
FROM flow_opd_runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive


RUN install -d -o 10001 -g 10001 /workspace
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner flow_opd" \
      dev.frontier.problem-id="flow_opd" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime

# Snapshot only public toolchain/model settings, never the build environment's
# credentials. SSH has a deliberately minimal environment.
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path('"'"'/opt/frontier/runtime-python'"'"'); output = path.open('"'"'x'"'"'); output.write('"'"'#!/bin/sh\nexec '"'"' + shlex.quote(sys.executable) + '"'"' "$@"\n'"'"'); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/repo/frontier_shim/accelerate

COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
