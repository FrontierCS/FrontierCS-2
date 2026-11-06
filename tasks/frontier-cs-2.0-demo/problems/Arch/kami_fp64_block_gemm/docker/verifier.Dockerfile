# Verifier control image of kami_fp64_block_gemm: the shared Verifier base plus the
# shared GPU adapter (gpu-tools), which runs the judge on one H100 in
# docker/verifier-gpu.Dockerfile. No judge code runs here. Build context: this task's root, with
# .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnss-wrapper openssh-client \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier control kami_fp64_block_gemm" \
      dev.frontier.problem-id="kami_fp64_block_gemm"

COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
