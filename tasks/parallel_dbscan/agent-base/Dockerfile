# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot: the buildpack-deps package set of the
# source's Debian base and Python's pip, venv and dev tooling, so that no tool the agent had goes
# away. The agent runs as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential git tar ca-certificates curl gnupg dirmngr netbase sq wget mercurial openssh-client subversion procps autoconf automake bzip2 dpkg-dev file g++ gcc imagemagick libtool libtool-bin m4 make patch pkg-config unzip xz-utils zip default-libmysqlclient-dev libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libwebp-dev libxml2-dev libxslt1-dev libyaml-dev zlib1g-dev tk-dev uuid-dev python3-dev python3-pip python3-venv python3-wheel python-is-python3 idle3 \
    && install --directory --owner=1000 --group=1000 /workspace
# The source's images compiled with GCC 12 (Debian 12.2.0); the shared Ubuntu base ships GCC 13,
# which
# measurably changed one historical submission's timings. GCC 12 is installed and linked as
# `gcc`/`g++`/`cc`/`c++`/`cpp` for the agent, the Makefile and the judge, fetched as above.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh gcc-12 g++-12 cpp-12 \
    && for tool in gcc g++ cpp gcov gcc-ar gcc-nm gcc-ranlib; do ln --symbolic --force "$tool-12" "/usr/bin/$tool"; done \
    && ln --symbolic --force gcc-12 /usr/bin/cc && ln --symbolic --force g++-12 /usr/bin/c++ \
    && test "$(g++ -dumpversion | cut -d. -f1)" = 12 && test "$(cc -dumpversion | cut -d. -f1)" = 12 \
    && test "$(c++ -dumpversion | cut -d. -f1)" = 12 && test "$(cpp -dumpversion | cut -d. -f1)" = 12

LABEL org.opencontainers.image.title="FrontierCS2 task parallel_dbscan" \
      dev.frontier.problem-id="parallel_dbscan"

# The solver-visible tree, and nothing else: the evaluator, oracles, reference and final seeds
# are in the verifier's image only. The submission tools are the repository's `shared/` ones,
# which the build puts in its context, and which replace the base's.
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

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
