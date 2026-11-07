# syntax=docker/dockerfile:1.7
# GPU Verifier image of flow_opd: the task's published offline runtime (the same digest-pinned
# base as docker/agent-gpu.Dockerfile and the previous release's Verifier, so every package
# and model file is unchanged), this task's judge and starter, and the shared verifier modules.
# The SSH bootstrap comes from the shared gpu-tools image.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM docker.io/joyemang33/fcs2-task-flow_opd-gpu-runtime@sha256:47687f9d3935b55d321303ada053de92f435c70ac5f497952fcc8e3e6f914da5

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier flow_opd" \
      dev.frontier.problem-id="flow_opd" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task; pristine/ is the starter the shared verifier applies the patch to
# (and the evaluator's runtime adapter).
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
# The shared verifier (edit_scope, then check_tree, then evaluate(tree)), from the shared
# Verifier base v10 by digest.
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml

# Grant read/traverse to both platform UIDs while keeping the bundle immutable and its
# executable entrypoints executable.
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path('"'"'/opt/frontier/runtime-python'"'"'); output = path.open('"'"'x'"'"'); output.write('"'"'#!/bin/sh\nexec '"'"' + shlex.quote(sys.executable) + '"'"' "$@"\n'"'"'); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0555 /opt/frontier/task/pristine/frontier_shim/accelerate
# The judge runs each workload with the shared isolate launcher (userns backend on the
# runner): own namespaces, no judge files, the GPU and the offline models.
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
# /opt/frontier on the judge interpreter's default import path (frontier.pth), so the judge and
# every worker process it starts import isolate by name.
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
