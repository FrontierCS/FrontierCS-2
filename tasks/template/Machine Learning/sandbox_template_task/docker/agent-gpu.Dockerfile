# syntax=docker/dockerfile:1.7
# Agent GPU runner image of sandbox_template_task, and the image of the untrusted command test
# sandbox [test_sandboxes.candidate]: the solver's runners and the judge's class-1 machines are
# the same environment. It holds the pinned CUDA/PyTorch image, the read-only starter and the
# task runtime; nothing of the judge (the judge uploads its fit driver with each run).
#
# Rules (scripts/validate_task_layout.py): as in tasks/template/MLSys/gpu_template_task; an
# untrusted sandbox's image is this judge-free one.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS, no default; the build injects tasks/shared/bases.json["gpu-tools"]); here
#     the SSH bootstrap the runner starts with;
#   - the toolchain image is pinned by digest; COPY only repo/ and gpu/task_runtime.py;
#   - the bootstrap starts as root and runs the solver's commands as frontier (uid 10001).
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client openssh-server tar \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace
LABEL org.opencontainers.image.title="FrontierCS2 sandbox_template_task Agent H100 runtime" \
      dev.frontier.problem-id="sandbox_template_task"
COPY repo /frontier/task/repo
COPY gpu/task_runtime.py /opt/frontier/task-runtime
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
# /opt/frontier/runtime-python is the image's Python; task-runtime replays its toolchain
# environment for commands over SSH (not a login shell).
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} +
USER root
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
