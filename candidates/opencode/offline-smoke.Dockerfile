# syntax=docker/dockerfile:1.7
ARG FRONTIER_TEST_OVERLAY_IMAGE
ARG FRONTIER_NODE_BUILD_IMAGE=node:22.23.3-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM ${FRONTIER_TEST_OVERLAY_IMAGE} AS candidate
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS fixture-node
FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
RUN useradd --uid 10001 --create-home frontier \
    && install -d -o 10001 -g 10001 /workspace /frontier/task /var/run/frontier \
    && printf '%s\n' 'Offline fixture: submit solution.txt in an uncompressed tar.' > /frontier/task/README.md \
    && printf '%s' 'offline-driver-capability' > /var/run/frontier/token \
    && chmod 0444 /var/run/frontier/token
COPY --from=candidate /opt/frontier/candidate /opt/frontier/candidate
# The overlay ships no Node; the fixture servers run on the build image's node.
COPY --from=fixture-node /usr/local/bin/node /fixture/node
COPY offline_container_smoke.mjs /fixture/smoke.mjs
USER 10001:10001
ENTRYPOINT ["/fixture/node", "/fixture/smoke.mjs"]
