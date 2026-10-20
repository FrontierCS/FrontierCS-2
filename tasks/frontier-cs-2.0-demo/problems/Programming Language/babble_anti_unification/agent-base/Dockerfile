# syntax=docker/dockerfile:1.7
# Task CPU base for babble_anti_unification. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
# Keep the authoritative compiler byte-identical while using the platform wrapper's
# supported libc runtime. Bullseye glibc cannot start the injected wrapper.
FROM rust:1.64-bullseye@sha256:53ded1c919ea0dc23be959e28037238d8c321cd88fbac04d818f210031fccc3b AS rust-toolchain
FROM debian:bookworm-slim@sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo \
    RUSTUP_TOOLCHAIN=1.64.0 PATH=/usr/local/cargo/bin:${PATH}

LABEL org.opencontainers.image.title="FrontierCS2 task babble_anti_unification" \
      dev.frontier.problem-id="babble_anti_unification" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_ble-popl23"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit


# The generic submit executable remains Candidate-owned. The compatibility link
# above also makes it discoverable when a tool shell resets PATH.
# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task
COPY agent-base/frontier/task/repo/README.md /frontier/task/repo/README.md

RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && if [ -f /frontier/task/bin/frontier-gpu ]; then \
         chmod 0555 /frontier/task/bin/frontier-gpu; \
         ln -s /frontier/task/bin/frontier-gpu /frontier/task/frontier-gpu; \
       fi \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /workspace
