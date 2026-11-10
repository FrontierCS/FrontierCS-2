# syntax=docker/dockerfile:1.7
# Verifier GPU image of sandbox_template_task: the image of the trusted command test sandbox
# [test_sandboxes.timing]. It holds the task's pinned CUDA/PyTorch toolchain (the same as the
# Agent GPU image's, so the times are the solver's environment) and the shared isolate launcher;
# the judge uploads its timing harness and the trees with each run, so no judge file is baked in.
#
# Rules (scripts/validate_task_layout.py):
#   - shared code only from shared stages, by build argument: the SSH bootstrap, isolate.py and
#     install_frontier_pth.py from `FROM ${GPU_TOOLS} AS frontier_gpu_tools`;
#   - a task with a command sandbox also declares `FROM ${VERIFIER_BASE} AS frontier_verifier`
#     (check_gpu_adapter_pins: the current shared verifier release); this image copies nothing
#     from it, since no verification runs here;
#   - frontier.pth puts /opt/frontier on the image Python's import path: the harness imports
#     isolate by name; on the machine (gVisor) isolate builds its userns backend;
#   - /opt/frontier/runtime-python is the interpreter the judge's steps name.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openssh-client openssh-server tar \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace
LABEL org.opencontainers.image.title="FrontierCS2 verifier sandbox_template_task H100 timing" \
      dev.frontier.problem-id="sandbox_template_task"
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import torch' \
    && chmod 0444 /opt/frontier/isolate.py \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /workspace
