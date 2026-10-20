# syntax=docker/dockerfile:1.7
# Verifier for approximate_mips_query_stopping: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier; nothing here may call Driver
# endpoints, reach S3, or assume a writable root filesystem.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 verifier approximate_mips_query_stopping" \
      dev.frontier.problem-id="approximate_mips_query_stopping" \
      dev.frontier.generator="frontiersmith2"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir zstandard  \
    && install -d -o 65532 -g 65532 /frontier

# The judge side of the task. `repo` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/repo
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py

RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a-w /opt/frontier/task

USER 65532:65532
ENV TMPDIR=/frontier HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
