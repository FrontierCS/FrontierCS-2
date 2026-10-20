# Preserve the qualified published CPU runtime; update only public instructions.
FROM docker.io/joyemang33/fcs2-task-flow_opd-agent@sha256:742c6f9b2e8ea4aebce1e2c5f96d027755fd8785426847f9ae910a14a8854115
USER root
COPY agent-base/frontier/task/README.md /frontier/task/README.md
COPY agent-base/frontier/task/repo/README.md /frontier/task/repo/README.md
RUN chmod 0444 /frontier/task/README.md /frontier/task/repo/README.md
USER 10001:10001
