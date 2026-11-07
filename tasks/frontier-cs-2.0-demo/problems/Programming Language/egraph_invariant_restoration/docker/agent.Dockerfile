# Agent image of egraph_invariant_restoration: the shared Agent base plus this task's packages, the
# Rust 1.94.1 toolchain the judge builds with, the statement, the starter repo and the generated
# submission policy. Composed with the Candidate overlay at Run time, so it must not depend on
# its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The old image was rust:1.94.1-bookworm (buildpack-deps): its manually installed packages are
# kept so the Agent's tools do not shrink (libncurses5-dev is libncurses-dev on Ubuntu 24.04).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 tar \
       autoconf automake bzip2 curl default-libmysqlclient-dev dpkg-dev file gnupg imagemagick \
       libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev \
       libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev \
       libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev \
       libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make \
       mercurial netbase openssh-client patch procps sq subversion unzip wget xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust 1.94.1 installed the way the rust:1.94.1-bookworm image (used before) installs it: rustup 1.29.0,
# minimal profile, under /usr/local; the rustup-init digest is that image's rustup binary.
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo \
    RUSTUP_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu RUST_VERSION=1.94.1 PATH=/usr/local/cargo/bin:${PATH}
RUN python3 -c 'import hashlib, sys, urllib.request; body = urllib.request.urlopen("https://static.rust-lang.org/rustup/archive/1.29.0/x86_64-unknown-linux-gnu/rustup-init").read(); assert hashlib.sha256(body).hexdigest() == "4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"; open("/tmp/rustup-init", "wb").write(body)' \
    && chmod 0755 /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.94.1 --default-host x86_64-unknown-linux-gnu \
    && rm /tmp/rustup-init \
    && chmod -R a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

LABEL org.opencontainers.image.title="FrontierCS2 task egraph_invariant_restoration" \
      dev.frontier.problem-id="egraph_invariant_restoration"

# The solver-visible tree, and nothing else: the evaluator, reference, correctness guards and the
# beat-reference specification are in the Verifier image only.
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
