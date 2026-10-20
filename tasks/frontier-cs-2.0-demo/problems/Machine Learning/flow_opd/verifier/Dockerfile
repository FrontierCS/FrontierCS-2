# syntax=docker/dockerfile:1.7
# Lightweight Kubernetes controller for GPU verifier flow_opd. The task
# evaluator and its offline CUDA/model layers live only in verifier-gpu; this
# image downloads the platform inputs and brokers one Driver-owned runner with
# the task-declared number of GPUs.
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 verifier control flow_opd" \
      dev.frontier.problem-id="flow_opd" \
      dev.frontier.generator="frontiersmith2"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates libnss-wrapper openssh-client python3 python3-zstandard \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o 65532 -g 65532 /frontier

COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py

RUN python3 -c 'import zstandard' \
    && printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && find /opt/frontier -maxdepth 1 -type f -name '*.py' -exec chmod 0444 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 \
    FRONTIER_VERIFIER_GPU_SECONDS=3000 \
    FRONTIER_VERIFIER_GPU_COUNT=1 \
    FRONTIER_MAX_SSH_BYTES_PER_DIRECTION=8388608
WORKDIR /frontier
