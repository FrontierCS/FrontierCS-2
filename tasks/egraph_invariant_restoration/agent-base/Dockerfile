ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot: the packages that the source's
# rust:1.94.1-bookworm base held, as the source restates them on Ubuntu. The agent runs as uid 1000,
# the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 tar autoconf automake bzip2 curl default-libmysqlclient-dev dpkg-dev file gnupg imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make mercurial netbase openssh-client patch procps sq subversion unzip wget xz-utils zlib1g-dev \
    && install --directory --owner=1000 --group=1000 /workspace
# Rust 1.94.1 as the source installs it, rustup 1.29.0 and the minimal profile under /usr/local, but
# with the rustup-init file and the installed toolchain each pinned by a digest (`fetch/rustup.py`).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUSTUP_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu RUST_VERSION=1.94.1 PATH=/usr/local/cargo/bin:${PATH}
RUN --mount=type=bind,source=fetch/rustup.py,target=/context/fetch/rustup.py \
    python3 -I /context/fetch/rustup.py 1.29.0 4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10 1.94.1 e7601339027f33d2dce8ec558d05b46a46e8e2c0c72dd30b247f64afc4f087f4 \
    && chmod --recursive a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

LABEL org.opencontainers.image.title="FrontierCS2 task egraph_invariant_restoration" \
      dev.frontier.problem-id="egraph_invariant_restoration"

# The solver-visible tree, and nothing else: the evaluator, reference and correctness guards are
# in the verifier's image only. The repository's submission tools replace the base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
