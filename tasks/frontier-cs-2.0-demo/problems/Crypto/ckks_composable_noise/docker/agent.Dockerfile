# Agent image of ckks_composable_noise: the shared Agent base plus this task's packages (Go
# 1.23.12 and the estimator's offline module cache), statement, starter repo and generated
# submission policy. Composed with the Candidate overlay at Run time, so it must not depend on
# its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git python3 tar time \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/frontier \
    && curl -fsSL https://go.dev/dl/go1.23.12.linux-amd64.tar.gz -o /opt/frontier/go.tar.gz \
    && echo 'd3847fef834e9db11bf64e3fb34db9c04db14e068eeb064f49af747010454f90  /opt/frontier/go.tar.gz' | sha256sum -c - \
    && tar -C /usr/local -xzf /opt/frontier/go.tar.gz && rm /opt/frontier/go.tar.gz \
    && ln -s /usr/local/go/bin/go /usr/local/bin/go && ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt

LABEL org.opencontainers.image.title="FrontierCS2 task ckks_composable_noise" \
      dev.frontier.problem-id="ckks_composable_noise"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

# The estimator's Go modules, offline (GOPROXY=off at run time).
RUN cd /frontier/task/repo/estimator \
    && GOMAXPROCS=1 GOTOOLCHAIN=local GOMODCACHE=/opt/frontier/go-mod /usr/local/go/bin/go mod download \
    && chmod -R a+rX /opt/frontier/go-mod /usr/local/go \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CKKS_MODULE_CACHE=/opt/frontier/go-mod GOMODCACHE=/opt/frontier/go-mod GOPROXY=off
WORKDIR /workspace
