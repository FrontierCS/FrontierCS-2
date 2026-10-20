# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
LABEL org.opencontainers.image.title="FrontierCS2 IVF-PQ GPU kernel agent base" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="ivf_pq_gpu_kernel_optimization" \
      dev.frontier.upstream-commit="1bccbadc6bead714d25c546cf954a37bb0791fd6"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client python3 tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace
COPY agent-base/frontier/task /frontier/task
COPY agent-base/frontier/task/repo/README.md /frontier/task/repo/README.md
RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh /frontier/task/repo/*.sh
USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100 FRONTIER_GPU_SECONDS=10800
WORKDIR /workspace
