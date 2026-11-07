# Agent image of parallel_dbscan: the shared Agent base (task.toml `base`) plus this
# task's packages, statement, starter repo and generated submission policy. Composed with the
# Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

# The old Agent image was python:3.11-bookworm (buildpack-deps:bookworm plus CPython and pip).
# The same buildpack-deps package set (curl, scm and full layers) and Python's pip/venv/dev
# tooling are installed here on the shared Ubuntu base, so no tool the Agent had goes away.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git tar \
       ca-certificates curl gnupg dirmngr netbase sq wget \
       mercurial openssh-client subversion procps \
       autoconf automake bzip2 dpkg-dev file g++ gcc imagemagick libtool libtool-bin m4 make patch \
       pkg-config unzip xz-utils zip \
       default-libmysqlclient-dev libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev \
       libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev \
       libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev \
       libreadline-dev libsqlite3-dev libssl-dev libwebp-dev libxml2-dev libxslt1-dev libyaml-dev \
       zlib1g-dev tk-dev uuid-dev \
       python3-dev python3-pip python3-venv python3-wheel python-is-python3 idle3 \
    && rm -rf /var/lib/apt/lists/*

# The old Agent and Verifier images compiled with GCC 12 (Debian 12.2.0); the shared Ubuntu base
# ships GCC 13, which measurably changed one historical submission's timings. GCC 12 is installed
# and linked as `gcc`/`g++`/`cc`/`c++`/`cpp` for the Agent, the Makefile and the judge.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc-12 g++-12 cpp-12 \
    && rm -rf /var/lib/apt/lists/* \
    && for tool in gcc g++ cpp gcov gcc-ar gcc-nm gcc-ranlib; do ln -sf "$tool-12" "/usr/bin/$tool"; done \
    && ln -sf gcc-12 /usr/bin/cc && ln -sf g++-12 /usr/bin/c++ \
    && test "$(g++ -dumpversion | cut -d. -f1)" = 12 && test "$(cc -dumpversion | cut -d. -f1)" = 12 \
    && test "$(c++ -dumpversion | cut -d. -f1)" = 12 && test "$(cpp -dumpversion | cut -d. -f1)" = 12

LABEL org.opencontainers.image.title="FrontierCS2 task parallel_dbscan" \
      dev.frontier.problem-id="parallel_dbscan"

# The solver-visible tree, and nothing else: the evaluator, oracles, reference and beat-reference
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
