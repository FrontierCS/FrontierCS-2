# syntax=docker/dockerfile:1.7
ARG FRONTIER_TEST_OVERLAY_IMAGE
FROM ${FRONTIER_TEST_OVERLAY_IMAGE} AS candidate
FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
RUN useradd --uid 10001 --create-home frontier \
    && install -d -o 10001 -g 10001 /workspace /frontier/task /var/run/frontier \
    && printf '%s\n' 'Offline fixture: submit solution.txt in an uncompressed tar.' > /frontier/task/README.md \
    && printf '%s' 'offline-driver-capability' > /var/run/frontier/token \
    && chmod 0444 /var/run/frontier/token
COPY --from=candidate /opt/frontier/candidate /opt/frontier/candidate
COPY offline_container_smoke.mjs /fixture/smoke.mjs
USER 10001:10001
ENTRYPOINT ["/opt/frontier/candidate/pi/node", "/fixture/smoke.mjs"]
