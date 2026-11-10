# Verifier control image of sandbox_template_task: the shared Verifier base, the shared GPU
# adapter, the judge and the starter. With a command test sandbox the placement is "control":
# the shared verifier applies the patch here and calls evaluate(tree), which leases the
# [test_sandboxes.*] machines through run_sandboxes() (fcs2_verifier_gpu.py) and scores here.
#
# Rules:
#   - the adapter from the shared gpu-tools stage (`FROM ${GPU_TOOLS} AS ...`), the final stage
#     FROM ${VERIFIER_BASE};
#   - the judge at /opt/frontier/task and the starter at pristine/, as in a CPU task;
#   - libnss-wrapper and openssh-client: the adapter's ssh to the machines as uid 65532.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libnss-wrapper openssh-client \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier control sandbox_template_task" \
      dev.frontier.problem-id="sandbox_template_task"

COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
