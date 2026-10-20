# syntax=docker/dockerfile:1.7
# Lightweight Kubernetes controller for GPU verifier mujoco_playground_cpu. The task
# evaluator and its offline CUDA/model layers live only in verifier-gpu; this
# image downloads the platform inputs and brokers one Driver-owned runner with
# the task-declared number of GPUs.
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 verifier control mujoco_playground_cpu" \
      dev.frontier.problem-id="mujoco_playground_cpu" \
      dev.frontier.generator="manual-authoring"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates libnss-wrapper openssh-client python3 \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o 65532 -g 65532 /frontier

COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py

RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && chmod 0444 /opt/frontier/*.py

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 \
    FRONTIER_VERIFIER_GPU_SECONDS=5400 \
    FRONTIER_VERIFIER_GPU_EVAL_SECONDS=14400 \
    FRONTIER_VERIFIER_GPU_COUNT=1 \
    FRONTIER_MAX_SSH_BYTES_PER_DIRECTION=8388608
WORKDIR /frontier
