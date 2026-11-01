# syntax=docker/dockerfile:1.7
# Task CPU base for detok_latent_denoising. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task detok_latent_denoising" \
      dev.frontier.problem-id="detok_latent_denoising" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="unknown"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client python3 tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
# GPU dependency assets are consumed by the GPU recipes; keep the CPU editor's
# original statement, helpers and repository without duplicating those assets.
COPY agent-base/frontier/task/README.md /frontier/task/README.md
COPY agent-base/frontier/task/bin /frontier/task/bin
COPY agent-base/frontier/task/repo /frontier/task/repo

RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV FRONTIER_GPU_TYPE=A100-80GB FRONTIER_GPU_SECONDS=5400
WORKDIR /workspace
