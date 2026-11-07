# syntax=docker/dockerfile:1.7
# GPU Verifier image of kmeans_gpu_kernel_optimization: the pinned PyTorch toolchain plus this
# task's judge and the shared verifier (edit_scope, then check_tree, then evaluate(tree)). The
# SSH bootstrap and the isolate launcher come from the shared gpu-tools image and the
# verifier modules from the shared Verifier base (tasks/shared/bases.json). Build context: this
# task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client openssh-server tar \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace /frontier
LABEL org.opencontainers.image.title="FrontierCS2 KMeans Verifier H100 runtime" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="kmeans_gpu_kernel_optimization"
# The judge, its generated config and the starter at pristine/ (what the patch applies to).
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
# The judge runs every implementation (naive, reference, candidate) with the shared isolate
# launcher (userns backend on the runner): own namespaces, the GPU, no judge files, nothing left
# for the next run.
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import torch, zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier/verifier /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod -R a+rX,a-w /opt/frontier/task
# /opt/frontier on the judge interpreter's default import path (frontier.pth), so the judge and
# every worker process it starts import isolate by name.
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
