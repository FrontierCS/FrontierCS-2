# Verifier control image of fast_kv_compaction: the shared Verifier base, the shared GPU adapter
# (gpu-tools) and this task's judge and starter. Placement "control" (task.toml
# [test_sandboxes.shard]): the shared verifier applies the patch and runs evaluate(tree) here;
# evaluate() measures on single-H100 runners from docker/verifier-gpu.Dockerfile (4 in train,
# 8 in final) through run_sandboxes().
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

LABEL org.opencontainers.image.title="FrontierCS2 verifier control fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction"

# The judge and the starter (evaluate() uploads both, with the applied tree, to every runner).
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
