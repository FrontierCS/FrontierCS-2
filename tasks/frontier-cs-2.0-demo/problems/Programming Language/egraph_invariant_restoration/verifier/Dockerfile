# syntax=docker/dockerfile:1.7
FROM rust:1.94.1-bookworm@sha256:4ec71e955e6c08aeb238885083222ddff79d82eb87654a96c76e38e94da1a53b
LABEL org.opencontainers.image.title="FrontierCS2 verifier egraph_invariant_restoration" \
      dev.frontier.problem-id="egraph_invariant_restoration" \
      dev.frontier.generator="manual-transfer"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir pyyaml zstandard \
    && install -d -o 65532 -g 65532 /frontier
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/
COPY verifier/beat_spec_feedback.py /opt/frontier/
COPY verifier/private_metrics.py /opt/frontier/
RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a-w /opt/frontier/task
USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
