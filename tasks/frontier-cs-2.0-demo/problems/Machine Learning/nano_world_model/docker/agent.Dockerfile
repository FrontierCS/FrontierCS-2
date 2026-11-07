# Agent image of nano_world_model: the shared Agent base (task.toml `base`) plus this task's packages,
# statement and starter repo. The GPU runner is docker/agent-gpu.Dockerfile. Composed with the
# Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client python3 tar \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task nano_world_model" \
      dev.frontier.problem-id="nano_world_model"

COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN chmod -R a+rX,a-w /frontier/task \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV FRONTIER_GPU_TYPE=A100-80GB FRONTIER_GPU_SECONDS=1800
WORKDIR /workspace
