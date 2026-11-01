# syntax=docker/dockerfile:1.7
# Task CPU base for bounded_evm_path_prediction. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task bounded_evm_path_prediction" \
      dev.frontier.problem-id="bounded_evm_path_prediction" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top75_6e3582446a"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git golang-go python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

RUN chmod 0555 /frontier/task/bin/*.sh \
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

# Go module cache for the repository's pinned modules. The Agent sandbox has no
# network access, so `go build` and `go test` resolve modules from this cache;
# GOPROXY serves the same cache, so a build with another GOMODCACHE also works.
USER 0:0
RUN cp -a /frontier/task/repo /tmp/modules \
    && cd /tmp/modules \
    && GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-modcacherw go mod download \
    && GOMODCACHE=/opt/frontier/go-mod GOPROXY=off GOCACHE=/tmp/go-build go build ./... \
    && GOMODCACHE=/opt/frontier/go-mod GOPROXY=off GOCACHE=/tmp/go-build go vet ./core/vm \
    && rm -rf /tmp/modules /tmp/go-build \
    && chmod -R a-w,a+rX /opt/frontier/go-mod

USER 10001:10001
ENV GOMODCACHE=/opt/frontier/go-mod GOPROXY=file:///opt/frontier/go-mod/cache/download GOSUMDB=off GOTOOLCHAIN=local
