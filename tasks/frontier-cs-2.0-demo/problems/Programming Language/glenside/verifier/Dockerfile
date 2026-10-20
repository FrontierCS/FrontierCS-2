# syntax=docker/dockerfile:1.7
# Verifier for glenside: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM rust:1.85.1-slim-bookworm@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS glenside_rust
FROM python:3.12-slim-bookworm@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a AS glenside_runtime
COPY --from=glenside_rust /usr/local/rustup /usr/local/rustup
COPY --from=glenside_rust /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY agent-base/frontier/task/repo/code /opt/frontier-cargo-input
RUN CARGO_HOME=/opt/cargo cargo fetch --locked --manifest-path /opt/frontier-cargo-input/Cargo.toml \
    && chmod -R a+rX /opt/cargo
ENV CARGO_HOME=/opt/cargo CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
COPY agent-base/frontier/task/repo /opt/frontier-public
RUN apt-get update && apt-get install -y --no-install-recommends patch build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 /opt/frontier-public/frontier-build.py --output /opt/frontier-prebuild --target-dir /opt/frontier-dependency-target \
    && CARGO_TARGET_DIR=/opt/frontier-dependency-target CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 \
       cargo test --manifest-path /opt/frontier-prebuild/code/Cargo.toml --locked --offline --no-default-features --lib --no-run \
    && cargo clean --manifest-path /opt/frontier-prebuild/code/Cargo.toml --target-dir /opt/frontier-dependency-target -p glenside \
    && chmod -R a+rX /opt/frontier-dependency-target

FROM glenside_runtime

USER root
# Cargo fingerprints include dependency source paths. Seed each private build
# at its runtime paths, then retain dependencies only (never a Glenside answer).
RUN set -eu; \
    for build_role in candidate trusted; do \
      export CARGO_HOME=/frontier/glenside-work/${build_role}-cargo-home; \
      export CARGO_TARGET_DIR=/frontier/glenside-work/${build_role}-target; \
      mkdir -p "${CARGO_HOME}"; \
      cp -a /opt/cargo/. "${CARGO_HOME}/"; \
      CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 \
        cargo test --manifest-path /opt/frontier-prebuild/code/Cargo.toml --locked --offline --no-default-features --lib --no-run; \
      cargo clean --manifest-path /opt/frontier-prebuild/code/Cargo.toml -p glenside; \
      mv "${CARGO_TARGET_DIR}" /opt/frontier-dependency-target-${build_role}; \
      chmod -R a+rX /opt/frontier-dependency-target-${build_role}; \
    done; \
    rm -rf /frontier/glenside-work

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates cargo git libseccomp2 patch python3 python3-pip rustc tar util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir pytest==9.1.1 pyyaml==6.0.2 zstandard


RUN install -d -o 65532 -g 65532 /frontier
# Jupyter and other NSS clients need an account for the numeric Verifier UID.
RUN if ! getent passwd 65532 >/dev/null; then useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier; fi
ENV HOME=/frontier

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier glenside" \
      dev.frontier.problem-id="glenside" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py


# Source snapshots are created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user, so merely removing write bits turns 0600 files into 0400 root-only
# files. Grant read/traverse while preserving executable files; the bundle
# stays immutable and becomes usable by both platform UIDs. Match the Agent's
# shell-entrypoint modes even when a sealed build context removed execute bits.
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path('"'"'/opt/frontier/runtime-python'"'"'); output = path.open('"'"'x'"'"'); output.write('"'"'#!/bin/sh\nexec '"'"' + shlex.quote(sys.executable) + '"'"' "$@"\n'"'"'); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && find /opt/frontier -maxdepth 1 -type f -name '*.py' -exec chmod 0444 {} + \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/harbor/app -type f -name '*.sh' -exec chmod 0555 {} +



USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GLENSIDE_WORK_ROOT=/frontier/glenside-work
WORKDIR /frontier
