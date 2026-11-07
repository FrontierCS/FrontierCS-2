# syntax=docker/dockerfile:1.7
# GPU Verifier image of dynamic_tanh: the task's toolchain plus the shared verifier modules and
# this task's judge. The SSH bootstrap comes from the shared gpu-tools image and the verifier
# modules from the shared verifier base (tasks/shared/bases.json); pip versions are held to the
# previous release's by docker/verifier-gpu.constraints.txt.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,source=docker/verifier-gpu.constraints.txt,target=/tmp/frontier-pip-constraints.txt \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-dev python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --constraint /tmp/frontier-pip-constraints.txt numpy==2.1.3 pillow==11.0.0 tensorboard==2.18.0 timm==1.0.15 torch==2.5.1 torchvision==0.20.1 zstandard

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier dynamic_tanh" \
      dev.frontier.problem-id="dynamic_tanh" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task: judge/ (evaluator, reference patch, beat spec, private flower
# workloads) and the starter repo as pristine/, which the shared verifier overlays with the
# submission's allowed files.
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
# The shared verifier (edit_scope, then check_tree, then evaluate(tree)), from the pinned base.
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml

# Source snapshots are created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user. Grant read/traverse while preserving executable files; the bundle
# stays immutable and becomes usable by both platform UIDs.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +
# The judge runs each training process with the shared isolate launcher (userns backend on
# the runner): its own namespaces, a copy of the tree, the workload's photographs, the GPU, no
# judge files and no inherited environment.
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
# /opt/frontier on the judge interpreter's default import path (frontier.pth), so the judge and
# every worker process it starts import isolate by name.
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
RUN chmod 0444 /opt/frontier/isolate.py
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
