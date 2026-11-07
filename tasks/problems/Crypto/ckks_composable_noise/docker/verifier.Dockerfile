# Verifier image of ckks_composable_noise, both modes: the shared Verifier base plus this task's
# packages (Go 1.23.12 and the estimator's offline module cache), judge bundle and generated
# config. The platform wrapper invokes /opt/frontier/verifier; task evaluator code never
# receives a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git python3 tar time \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/frontier \
    && curl -fsSL https://go.dev/dl/go1.23.12.linux-amd64.tar.gz -o /opt/frontier/go.tar.gz \
    && echo 'd3847fef834e9db11bf64e3fb34db9c04db14e068eeb064f49af747010454f90  /opt/frontier/go.tar.gz' | sha256sum -c - \
    && tar -C /usr/local -xzf /opt/frontier/go.tar.gz && rm /opt/frontier/go.tar.gz \
    && ln -s /usr/local/go/bin/go /usr/local/bin/go && ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt

LABEL org.opencontainers.image.title="FrontierCS2 verifier ckks_composable_noise" \
      dev.frontier.problem-id="ckks_composable_noise"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN cd /opt/frontier/task/pristine/estimator \
    && GOMAXPROCS=1 GOTOOLCHAIN=local GOMODCACHE=/opt/frontier/go-mod /usr/local/go/bin/go mod download \
    && chmod -R a+rX /opt/frontier/go-mod /usr/local/go \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GOMODCACHE=/opt/frontier/go-mod GOPROXY=off
WORKDIR /frontier
