# Verifier image of quarter_pir, both modes: the shared Verifier base plus this task's packages
# (Go 1.19.13 and the module cache, build-essential, python3-yaml), the judge bundle and
# generated config. The platform wrapper invokes /opt/frontier/verifier; task evaluator code
# never receives a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git python3 python3-yaml \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://go.dev/dl/go1.19.13.linux-amd64.tar.gz -o /tmp/go.tar.gz \
    && echo '4643d4c29c55f53fa0349367d7f1bb5ca554ea6ef528c146825b0f8464e2e668  /tmp/go.tar.gz' | sha256sum -c - \
    && tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz

LABEL org.opencontainers.image.title="FrontierCS2 verifier quarter_pir" \
      dev.frontier.problem-id="quarter_pir"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference and baseline patches to); its source is repo/ itself.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN cd /opt/frontier/task/pristine/quarter_pir/code \
    && GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-mod=mod /usr/local/go/bin/go mod download \
    && chmod -R a+rX /opt/frontier/go-mod \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GOPATH=/go GOMODCACHE=/opt/frontier/go-mod
WORKDIR /frontier
