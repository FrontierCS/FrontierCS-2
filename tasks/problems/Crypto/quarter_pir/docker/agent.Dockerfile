# Agent image of quarter_pir: the shared Agent base plus this task's packages (Go 1.19.13, the
# module cache served offline as GOPROXY, and the tools the old golang:1.19-based image had:
# build-essential, curl, wget, gnupg, mercurial, subversion, openssh-client, pkg-config,
# python3-yaml, python3-pytest), statement, starter repo and generated submission policy.
# Composed with the Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT,
# CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git gnupg mercurial openssh-client pkg-config python3 python3-pytest python3-yaml subversion wget \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://go.dev/dl/go1.19.13.linux-amd64.tar.gz -o /tmp/go.tar.gz \
    && echo '4643d4c29c55f53fa0349367d7f1bb5ca554ea6ef528c146825b0f8464e2e668  /tmp/go.tar.gz' | sha256sum -c - \
    && tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz

LABEL org.opencontainers.image.title="FrontierCS2 task quarter_pir" \
      dev.frontier.problem-id="quarter_pir"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

# The modules of the pinned go.mod/go.sum, offline.
RUN cd /frontier/task/repo/quarter_pir/code \
    && GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-mod=mod /usr/local/go/bin/go mod download \
    && chmod -R a+rX /opt/frontier/go-mod \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GOLANG_VERSION=1.19.13 GOPATH=/go GOMODCACHE=/opt/frontier/go-mod \
    GOPROXY=file:///opt/frontier/go-mod/cache/download GOSUMDB=off
WORKDIR /workspace
