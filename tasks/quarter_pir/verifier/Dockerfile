# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image; then Go 1.19.13 from go.dev, checked by
# its SHA-256.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates curl git python3 python3-yaml \
    && curl --fail --silent --show-error --location --output /tmp/go.tar.gz \
       https://go.dev/dl/go1.19.13.linux-amd64.tar.gz \
    && echo '4643d4c29c55f53fa0349367d7f1bb5ca554ea6ef528c146825b0f8464e2e668  /tmp/go.tar.gz' \
       | sha256sum --check - \
    && tar --extract --gzip --file=/tmp/go.tar.gz --directory=/usr/local \
    && rm /tmp/go.tar.gz

LABEL org.opencontainers.image.title="FrontierCS2 verifier quarter_pir" \
      dev.frontier.problem-id="quarter_pir"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier, and the launcher through which the judge runs every process it builds
# from a submission: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The task's judge, with the starter repo it trusts at pristine/: the trusted tools and the
# reference are built from there, never from the submission. And each mode's plan: the train
# judge reads train.tar, and the final judge eval.tar.
COPY --chmod=a+rX,a-w judge /opt/frontier/task
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.tar datasets/eval.tar /opt/frontier/datasets/
# The modules go.sum pins, checked against go.sum alone, as in the agent's image.
RUN cd /opt/frontier/task/pristine/quarter_pir/code \
    && export GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-mod=readonly GOSUMDB=off \
    && GOPROXY=https://proxy.golang.org /usr/local/go/bin/go mod download \
    && GOPROXY=off /usr/local/go/bin/go list -deps ./... > /dev/null \
    && chmod --recursive a+rX /opt/frontier/go-mod \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GOPATH=/go GOMODCACHE=/opt/frontier/go-mod
WORKDIR /frontier
