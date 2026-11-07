# Verifier image of glenside, both modes: the shared Verifier base plus this task's toolchain
# (Rust 1.85.1 as the pinned rust image installs it, the offline Cargo closure, the dependency
# build seed), packages, judge bundle and generated config. The platform
# wrapper invokes /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or
# Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates cargo git patch python3 python3-pip rustc tar util-linux xz-utils \
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
ENV CARGO_HOME=/opt/cargo CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
COPY repo /opt/frontier-public
RUN python3 /opt/frontier-public/frontier-build.py --output /opt/frontier-prebuild --target-dir /opt/frontier-dependency-target \
    && rm -rf /opt/frontier-dependency-target /opt/frontier-cargo-input /opt/frontier-public

# The dependency build seed every judge build copies into its isolate instance. Cargo
# fingerprints include the dependency source paths, so it is built against the CARGO_HOME the
# instances read in place (/opt/cargo); the target dir itself may move. Dependencies only
# (never a Glenside answer).
RUN CARGO_TARGET_DIR=/opt/frontier-cargo-seed CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 \
      cargo test --manifest-path /opt/frontier-prebuild/code/Cargo.toml --locked --offline --no-default-features --lib --no-run \
    && CARGO_TARGET_DIR=/opt/frontier-cargo-seed \
      cargo clean --manifest-path /opt/frontier-prebuild/code/Cargo.toml -p glenside \
    && chmod -R a+rX,a-w /opt/frontier-cargo-seed /opt/cargo \
    && install -d -o 65532 -g 65532 /frontier

LABEL org.opencontainers.image.title="FrontierCS2 verifier glenside" \
      dev.frontier.problem-id="glenside"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GLENSIDE_WORK_ROOT=/frontier/glenside-work
WORKDIR /frontier
