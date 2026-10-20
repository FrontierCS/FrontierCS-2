# syntax=docker/dockerfile:1.7
# Lightweight Kubernetes controller for GPU verifier fast_kv_compaction. The task
# evaluator and its offline CUDA/model layers live only in verifier-gpu; this
# image downloads the platform inputs and brokers one Driver-owned runner with
# the task-declared number of GPUs.
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 verifier control fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction" \
      dev.frontier.generator="frontiersmith2"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates git libnss-wrapper openssh-client python3 \
    && cd /tmp \
    && apt-get download python3-zstandard=0.22.0-1build1 \
    && echo 'b1e559a3754fd5108ec1dd6eaf4c88a6b597557410f2eaf46b1e464f5614db75  python3-zstandard_0.22.0-1build1_amd64.deb' | sha256sum -c - \
    && apt-get install -y --no-install-recommends ./python3-zstandard_0.22.0-1build1_amd64.deb \
    && rm python3-zstandard_0.22.0-1build1_amd64.deb \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c 'import zstandard' \
    && install -d -o 65532 -g 65532 /frontier

COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/gpu_fanout.py /opt/frontier/gpu_fanout.py
# The judge (patch checks, frozen targets, gate) runs here; measurement fans out
# to single-H100 Verifier GPU instances that hold the models.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app

RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && chmod 0555 /opt/frontier/verifier

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_VERIFIER_GPU_TYPE=H100 \
    FRONTIER_VERIFIER_GPU_SECONDS=5400 \
    FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
