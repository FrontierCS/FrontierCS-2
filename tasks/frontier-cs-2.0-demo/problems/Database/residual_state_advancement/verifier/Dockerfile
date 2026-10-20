# syntax=docker/dockerfile:1.7
# Verifier for residual_state_advancement: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier; nothing here may call Driver
# endpoints, reach S3, or assume a writable root filesystem.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM python:3.12-slim-bookworm@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a

LABEL org.opencontainers.image.title="FrontierCS2 verifier residual_state_advancement" \
      dev.frontier.problem-id="residual_state_advancement" \
      dev.frontier.generator="frontiersmith2"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir zstandard duckdb==1.5.1 pandas==2.3.3 pyyaml==6.0.2 scikit-learn==1.7.2 \
    && install -d -o 65532 -g 65532 /frontier

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/private_metrics.py /opt/frontier/private_metrics.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py

RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX /opt/frontier \
    && chmod -R a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
