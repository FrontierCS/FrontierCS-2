# syntax=docker/dockerfile:1.7
# Task CPU base for sparse_video_attention_policy. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself. The research statement is
# included under agent-base/frontier/task/repo/
# and therefore reaches /frontier/task/repo/README.md with the public tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task sparse_video_attention_policy" \
      dev.frontier.problem-id="sparse_video_attention_policy" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_82_svg_ear_v12"

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

# COPY preserves restrictive export modes, including helper parent directories.
RUN chmod -R a+rX,a-w /frontier/task \
    && chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV FRONTIER_GPU_TYPE=H100 FRONTIER_GPU_SECONDS=18000
WORKDIR /workspace
