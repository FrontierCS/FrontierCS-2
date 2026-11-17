# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot: the package set of buildpack-deps:noble,
# which the source's images were built on before, plus cmake and python3. The agent runs as uid
# 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh autoconf automake bzip2 ca-certificates cmake curl default-libmysqlclient-dev dpkg-dev file g++ gcc git gnupg imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make mercurial netbase openssh-client patch procps python3 sq subversion unzip wget xz-utils zlib1g-dev \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task evm_state_checkpoints" \
      dev.frontier.problem-id="evm_state_checkpoints"

# The solver-visible tree, and nothing else: the evaluator, reference and final seeds are in
# the verifier's image only. The submission tools are the repository's `shared/` ones, which
# the build puts in its context, and which replace the base's.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md
# Spectrum's CMake project clones nine dependencies from GitHub, each by a tag; here each tag is
# read from
# a mirror of the commit pinned in `github-pins.txt` (`fetch/git_mirror.sh`), which the step
# removes after.
# intx, one of them, makes Hunter download Hunter, GTest and benchmark: `hunter-pins.txt` names
# each with
# its SHA-256, and `fetch/download.sh` puts them where Hunter looks, so that it finds them and
# fetches
# nothing. The configure has no route to HTTPS but to raw.githubusercontent.com, whence intx reads
# HunterGate.cmake, which its own SHA-256 checks, so that a fetch outside the pins fails the build.
# The Agent sandbox has no network, so the configured dependency tree is published read-only at
# /opt/frontier/spectrum-deps.
RUN --mount=type=bind,source=fetch/git_mirror.sh,target=/context/fetch/git_mirror.sh \
    --mount=type=bind,source=fetch/download.sh,target=/context/fetch/download.sh \
    --mount=type=bind,source=github-pins.txt,target=/context/github-pins.txt \
    --mount=type=bind,source=hunter-pins.txt,target=/context/hunter-pins.txt \
    sh /context/fetch/git_mirror.sh /context/github-pins.txt /opt/frontier-pins \
    && sh /context/fetch/download.sh /context/hunter-pins.txt /workspace/.hunter \
    && install --directory --owner=1000 --group=1000 /workspace \
    && cp --archive /frontier/task/repo /workspace/repo \
    && chmod --recursive u+w /workspace/repo \
    && HOME=/workspace https_proxy=http://127.0.0.1:9 no_proxy=raw.githubusercontent.com \
       cmake -S /workspace/repo -B /workspace/repo/build -DNDEBUG=ON >/dev/null \
    && install --directory /opt/frontier \
    && mv /workspace/repo/build/_deps /opt/frontier/spectrum-deps \
    && rm --recursive --force /workspace/repo /workspace/.cmake /workspace/.hunter /root/.hunter \
    && chmod --recursive a+rX,a-w /opt/frontier/spectrum-deps \
    && git config --system --unset-all 'url.file:///opt/frontier-pins/.insteadOf' && rm --recursive --force /opt/frontier-pins

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
