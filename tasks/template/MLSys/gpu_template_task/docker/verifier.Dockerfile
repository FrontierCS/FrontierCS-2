# Verifier control image of gpu_template_task: the shared Verifier base plus the shared GPU
# adapter. With placement "runner" (task.toml: [gpu.verifier] and no command test sandbox) no
# judge code runs here: the shared verifier ships the payload and dataset to one runner of
# docker/verifier-gpu.Dockerfile and copies its result back.
#
# Rules:
#   - the adapter comes from the shared gpu-tools stage (`FROM ${GPU_TOOLS} AS ...`); the final
#     stage is FROM ${VERIFIER_BASE};
#   - FRONTIER_VERIFIER_GPU_TYPE and FRONTIER_VERIFIER_GPU_COUNT name the runner (the Driver
#     leases quota.eval_timeout_seconds, or [gpu.verifier] lease_seconds, per mode);
#     FRONTIER_VERIFIER_GPU_SECONDS is retired;
#   - libnss-wrapper and openssh-client: the adapter's ssh to the runner as uid 65532.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnss-wrapper openssh-client \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier control gpu_template_task" \
      dev.frontier.problem-id="gpu_template_task"

COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
