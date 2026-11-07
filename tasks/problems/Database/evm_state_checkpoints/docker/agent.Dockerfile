# Agent image of evm_state_checkpoints: the shared Agent base plus this task's packages,
# statement, starter repo and generated submission policy. Composed with the Candidate overlay
# at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

# The package set of buildpack-deps:noble, which the task's images were built on before
#, plus cmake and python3: the same compilers, build tools and -dev libraries.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       autoconf automake bzip2 ca-certificates cmake curl default-libmysqlclient-dev dpkg-dev file g++ gcc git gnupg imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make mercurial netbase openssh-client patch procps python3 sq subversion unzip wget xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task evm_state_checkpoints" \
      dev.frontier.problem-id="evm_state_checkpoints"

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

# Spectrum's CMake project fetches pinned dependencies; the Agent sandbox has no network, so
# the configured dependency tree is published read-only at /opt/frontier/spectrum-deps.
RUN install -d -o 10001 -g 10001 /workspace \
    && cp -a /frontier/task/repo /workspace/repo \
    && chmod -R u+w /workspace/repo \
    && HOME=/workspace cmake -S /workspace/repo -B /workspace/repo/build -DNDEBUG=ON >/dev/null \
    && install -d /opt/frontier \
    && mv /workspace/repo/build/_deps /opt/frontier/spectrum-deps \
    && rm -rf /workspace/repo /workspace/.cmake /workspace/.hunter /root/.hunter \
    && chmod -R a+rX,a-w /opt/frontier/spectrum-deps

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
