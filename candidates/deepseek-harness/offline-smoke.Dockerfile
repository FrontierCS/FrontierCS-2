# syntax=docker/dockerfile:1.7
ARG FRONTIER_TEST_OVERLAY_IMAGE
ARG FRONTIER_TEST_TASK_IMAGE
FROM ${FRONTIER_TEST_OVERLAY_IMAGE} AS candidate
FROM ${FRONTIER_TEST_TASK_IMAGE}
USER root
COPY --from=candidate /opt/frontier/candidate /opt/frontier/candidate
COPY offline_container_smoke.mjs /fixture/smoke.mjs
RUN mkdir -p /frontier/task /var/run/frontier /workspace \
    && printf '%s\n' 'Offline fixture: submit solution.txt in an uncompressed tar.' > /frontier/task/README.md \
    && printf '%s' 'offline-driver-capability' > /var/run/frontier/token \
    && chmod 0444 /var/run/frontier/token
USER 10001:10001
ENTRYPOINT ["/opt/frontier/candidate/deepseek-harness/node", "/fixture/smoke.mjs"]
