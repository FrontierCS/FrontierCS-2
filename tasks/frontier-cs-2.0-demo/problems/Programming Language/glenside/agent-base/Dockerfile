# syntax=docker/dockerfile:1.7
# Task CPU base for glenside. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
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

FROM glenside_runtime

LABEL org.opencontainers.image.title="FrontierCS2 task glenside" \
      dev.frontier.problem-id="glenside" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_2105-09377"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates cargo git libseccomp2 patch python3 rustc tar util-linux xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir pytest==9.1.1 pyyaml==6.0.2

# Login shells retain /usr/local/bin but reset the Cargo-specific PATH entry.
RUN for tool in cargo rustc rustdoc rustup; do \
      ln -s "/usr/local/cargo/bin/$tool" "/usr/local/bin/$tool"; \
    done

# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task
COPY agent-base/frontier/task/repo/code /opt/frontier-cargo-input
RUN diff -qr /frontier/task/repo/code /opt/frontier-cargo-input \
    && ! grep -q "^mod tests" /frontier/task/repo/code/src/language/language.rs

RUN chmod 0555 /frontier/task/bin/*.sh \
    && chmod 0555 /frontier \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task -type f -exec chmod a+r {} + \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CARGO_HOME=/workspace/.cargo

WORKDIR /workspace
