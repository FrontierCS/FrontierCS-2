# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
LABEL org.opencontainers.image.title="FrontierCS2 KMeans GPU verifier control" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="kmeans_gpu_kernel_optimization" \
      dev.frontier.upstream-commit="1f99d1f0700606908a5c4f1c6c11feba642ae0e5"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libnss-wrapper openssh-client python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-create-home --uid 65532 --shell /usr/sbin/nologin frontier-verifier \
    && install -d -o 65532 -g 65532 /frontier
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier
USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_SECONDS=1800 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
