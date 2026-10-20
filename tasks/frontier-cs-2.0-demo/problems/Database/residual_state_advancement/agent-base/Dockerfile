# syntax=docker/dockerfile:1.7
# Task CPU base for residual_state_advancement. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM python:3.12-slim-bookworm@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a

LABEL org.opencontainers.image.title="FrontierCS2 task residual_state_advancement" \
      dev.frontier.problem-id="residual_state_advancement" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top46_ef4c4c1bef"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git python3 tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir duckdb==1.5.1 pandas==2.3.3 pyyaml==6.0.2 scikit-learn==1.7.2


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

COPY agent-base/frontier/task/guide/README.md /frontier/task/repo/README.md
RUN chmod -R a+rX /frontier/task \
    && chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && if [ -f /frontier/task/bin/frontier-gpu ]; then \
         chmod 0555 /frontier/task/bin/frontier-gpu; \
         ln -s /frontier/task/bin/frontier-gpu /frontier/task/frontier-gpu; \
       fi \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
