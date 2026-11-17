# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages. Then Go 1.19.13 from go.dev, checked by its SHA-256. The agent runs as uid
# 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates curl git gnupg mercurial openssh-client \
           pkg-config python3 python3-pytest python3-yaml subversion wget \
    && curl --fail --silent --show-error --location --output /tmp/go.tar.gz \
       https://go.dev/dl/go1.19.13.linux-amd64.tar.gz \
    && echo '4643d4c29c55f53fa0349367d7f1bb5ca554ea6ef528c146825b0f8464e2e668  /tmp/go.tar.gz' \
       | sha256sum --check - \
    && tar --extract --gzip --file=/tmp/go.tar.gz --directory=/usr/local \
    && rm /tmp/go.tar.gz \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task quarter_pir" \
      dev.frontier.problem-id="quarter_pir"

# The solver-visible tree, and nothing else: the evaluator and reference are in the verifier's
# image only. The submission tools are the repository's `shared/` ones, which the build puts
# in its context, and which replace the base's.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

# The modules go.sum pins, from Go's module proxy, checked against go.sum alone: no checksum
# database vouches for one, go.mod and go.sum stay as they are, and listing the build's
# packages fails on a module go.sum lacks, which the download takes unchecked. The agent's
# builds read them offline.
RUN cd /frontier/task/repo/quarter_pir/code \
    && export GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-mod=readonly GOSUMDB=off \
    && GOPROXY=https://proxy.golang.org /usr/local/go/bin/go mod download \
    && GOPROXY=off /usr/local/go/bin/go list -deps ./... > /dev/null \
    && chmod --recursive a+rX /opt/frontier/go-mod \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GOLANG_VERSION=1.19.13 GOPATH=/go GOMODCACHE=/opt/frontier/go-mod \
    GOPROXY=file:///opt/frontier/go-mod/cache/download GOSUMDB=off
WORKDIR /workspace
