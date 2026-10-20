# syntax=docker/dockerfile:1.7
# Task CPU base for r_top01_2604-28190-cand_000-preflight-8250ab0a4a09. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM python:3.11-slim@sha256:174bec68e0451bffabbb08c7d5d21c6b253f772d81d52b9558af97bb3159b761

LABEL org.opencontainers.image.title="FrontierCS2 task r_top01_2604-28190-cand_000-preflight-8250ab0a4a09" \
      dev.frontier.problem-id="r_top01_2604-28190-cand_000-preflight-8250ab0a4a09" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_2604.28190"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client python3 tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task
COPY agent-base/frontier/task/repo/README.md /frontier/task/repo/README.md

RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task -type f -exec chmod 0444 {} + \
    && find /frontier/task -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV FRONTIER_GPU_TYPE=A100-80GB FRONTIER_GPU_SECONDS=7200
WORKDIR /workspace
