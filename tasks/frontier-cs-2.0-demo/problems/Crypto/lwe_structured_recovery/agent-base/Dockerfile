# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

LABEL org.opencontainers.image.title="FrontierCS2 structured-LWE agent base" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="lwe_structured_recovery" \
      dev.frontier.upstream-commit="15ce5cc28f69a79bc3555bc4c8de65424360b77f" \
      dev.frontier.catalog-sha256="318f32b68e8a30c6bcd5867c89eb10d87a5472614401f4e6df0f55356cdc98a6"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates fplll-tools git libgmp-dev libmpfr-dev \
       pkg-config python3 python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace

COPY agent-base/frontier/task /frontier/task

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/add_solution.py /frontier/task/bin/*.sh

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
