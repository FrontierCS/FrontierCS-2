# Agent image of glenside: the shared Agent base plus this task's toolchain (Rust 1.85.1 as the
# pinned rust image installs it, the offline Cargo dependency closure of repo/code, the apt and pip
# packages the task always had), statement, starter repo and generated submission policy.
# Composed with the Candidate overlay at Run time, so it must not depend on its own
# ENTRYPOINT, CMD or named user. Build context: this task's root, with .frontier-generated/
# added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates cargo git libseccomp2 patch python-is-python3 python3 python3-pip rustc tar util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir pytest==9.1.1 pyyaml==6.0.2

# Rust 1.85.1, installed exactly as the pinned rust:1.85.1-slim-bookworm image installs it (the
# task's toolchain before the migration): rustup-init 1.28.1 by checksum, minimal profile.
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUST_VERSION=1.85.1 \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
RUN python3 -c 'import urllib.request; urllib.request.urlretrieve("https://static.rust-lang.org/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init", "/tmp/rustup-init")' \
    && echo "a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f */tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host x86_64-unknown-linux-gnu \
    && rm /tmp/rustup-init \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME \
    && rustc --version
COPY repo/code /opt/frontier-cargo-input
RUN CARGO_HOME=/opt/cargo cargo fetch --locked --manifest-path /opt/frontier-cargo-input/Cargo.toml \
    && chmod -R a+rX /opt/cargo
# The old task's own prepare_workspace.sh copied /opt/cargo to /workspace/.cargo; the shared one
# does not. A vendored copy of the same locked dependencies, named in /.cargo/config.toml (which
# cargo reads from any working directory), keeps offline builds working with an empty CARGO_HOME.
RUN cd /opt/frontier-cargo-input \
    && mkdir -p /.cargo \
    && CARGO_HOME=/opt/cargo cargo vendor --locked --offline /opt/cargo-vendor > /.cargo/config.toml \
    && chmod -R a+rX /opt/cargo-vendor /.cargo
ENV CARGO_HOME=/opt/cargo CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
# Login shells retain /usr/local/bin but reset the Cargo-specific PATH entry.
RUN for tool in cargo rustc rustdoc rustup; do \
      ln -s "/usr/local/cargo/bin/$tool" "/usr/local/bin/$tool"; \
    done

LABEL org.opencontainers.image.title="FrontierCS2 task glenside" \
      dev.frontier.problem-id="glenside"

# The solver-visible tree, and nothing else: the evaluator, reference, private tests and
# beat-reference specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
RUN diff -qr /frontier/task/repo/code /opt/frontier-cargo-input \
    && ! grep -q "^mod tests" /frontier/task/repo/code/src/language/language.rs

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CARGO_HOME=/workspace/.cargo
WORKDIR /workspace
