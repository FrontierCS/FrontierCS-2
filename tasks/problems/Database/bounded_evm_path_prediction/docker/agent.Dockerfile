# Agent image of bounded_evm_path_prediction: the shared Agent base (task.toml `base`) plus this
# task's packages, statement, starter repo and generated submission policy. Composed with the
# Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git golang-go python3 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task bounded_evm_path_prediction" \
      dev.frontier.problem-id="bounded_evm_path_prediction"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace

# Go module cache for the repository's pinned modules (go.sum). The Agent sandbox has no
# network access, so `go build` and `go test` resolve modules from this cache; GOPROXY serves
# the same cache, so a build with another GOMODCACHE also works.
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
