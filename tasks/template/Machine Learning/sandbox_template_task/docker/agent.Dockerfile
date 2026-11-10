# Agent image of sandbox_template_task: the shared Agent base plus the statement, the starter repo
# and the generated submission policy, as in tasks/template/Theory/template_task. The Agent
# itself has no GPU: it leases runners of docker/agent-gpu.Dockerfile with frontier-gpu
# (task.toml [gpu.agent]). Build context: this task's root, with .frontier-generated/ added by
# the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-client python3 tar \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task sandbox_template_task" \
      dev.frontier.problem-id="sandbox_template_task"

COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
# The GPU the Agent's runners have, and their total budget (task.toml [gpu.agent]).
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100 FRONTIER_GPU_SECONDS=3600
WORKDIR /workspace
