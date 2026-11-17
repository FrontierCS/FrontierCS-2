# The Go toolchain and the estimator's module cache, built once for the agent's image and the
# verifier's, which copy them in. It builds on FrontierCS's shared verifier base, pinned by
# digest: Ubuntu 24.04 with apt set to one dated archive snapshot, and to try each file up to 10
# times.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# Then Go 1.23.12 from go.dev, checked by its SHA-256.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates curl \
    && curl --fail --silent --show-error --location --output /tmp/go.tar.gz \
       https://go.dev/dl/go1.23.12.linux-amd64.tar.gz \
    && echo 'd3847fef834e9db11bf64e3fb34db9c04db14e068eeb064f49af747010454f90  /tmp/go.tar.gz' \
       | sha256sum --check - \
    && tar --extract --gzip --file=/tmp/go.tar.gz --directory=/usr/local \
    && rm /tmp/go.tar.gz

# The modules go.sum pins, from Go's module proxy, checked against go.sum alone: no checksum
# database vouches for one, go.mod and go.sum stay as they are, and listing the estimator's
# packages fails on a module go.sum lacks, which the download takes unchecked.
COPY repo/estimator /src/estimator
RUN cd /src/estimator \
    && export GOMAXPROCS=1 GOTOOLCHAIN=local GOMODCACHE=/opt/frontier/go-mod \
       GOFLAGS=-mod=readonly GOSUMDB=off \
    && GOPROXY=https://proxy.golang.org /usr/local/go/bin/go mod download \
    && GOPROXY=off /usr/local/go/bin/go list -deps ./... > /dev/null \
    && chmod --recursive a+rX /opt/frontier/go-mod /usr/local/go \
    && rm --recursive --force /src
