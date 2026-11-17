# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY agent-base/requirements.txt /tmp/requirements.txt
# The task's packages, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000, the base
# image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bash build-essential ca-certificates cargo git libseccomp2 patch python-is-python3 python3 python3-pip rustc tar util-linux xz-utils \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=1000 --group=1000 /workspace
# Rust 1.85.1 as the source installs it, rustup 1.28.1 and the minimal profile under /usr/local, but
# with the rustup-init file and the installed toolchain each pinned by a digest (`fetch/rustup.py`).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUST_VERSION=1.85.1 PATH=/usr/local/cargo/bin:${PATH}
RUN --mount=type=bind,source=fetch/rustup.py,target=/context/fetch/rustup.py \
    python3 -I /context/fetch/rustup.py 1.28.1 a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f 1.85.1 638f4b5eb8d14aee8a3a1a4e3c9410c6a40bfb37452d7a4eec000b8ce12a4f69 \
    && chmod --recursive a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version
# The Cargo dependencies of the starter, offline from here on: `--locked` makes Cargo fetch what
# `Cargo.lock` names, each crates.io package checked against the lock's checksum and each git
# dependency taken at the lock's commit.
COPY repo/code /opt/frontier-cargo-input
RUN CARGO_HOME=/opt/cargo cargo fetch --locked --manifest-path /opt/frontier-cargo-input/Cargo.toml \
    && chmod --recursive a+rX /opt/cargo
# A vendored copy of the same locked dependencies, named in /.cargo/config.toml (which cargo
# reads from any working directory), keeps offline builds working with an empty CARGO_HOME.
RUN cd /opt/frontier-cargo-input \
    && mkdir --parents /.cargo \
    && CARGO_HOME=/opt/cargo cargo vendor --locked --offline /opt/cargo-vendor > /.cargo/config.toml \
    && chmod --recursive a+rX /opt/cargo-vendor /.cargo
ENV CARGO_HOME=/opt/cargo CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
# Login shells retain /usr/local/bin but reset the Cargo-specific PATH entry.
RUN for tool in cargo rustc rustdoc rustup; do \
      ln --symbolic "/usr/local/cargo/bin/$tool" "/usr/local/bin/$tool"; \
    done

LABEL org.opencontainers.image.title="FrontierCS2 task glenside" \
      dev.frontier.problem-id="glenside"

# The solver-visible tree, and nothing else: the evaluator, reference, private tests and the
# final workloads are in the verifier's image only. The submission tools are the repository's
# `shared/` ones, which the build puts in its context, and which replace the base's.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo
RUN diff --brief --recursive /frontier/task/repo/code /opt/frontier-cargo-input \
    && ! grep --quiet '^mod tests' /frontier/task/repo/code/src/language/language.rs

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CARGO_HOME=/workspace/.cargo
WORKDIR /workspace
