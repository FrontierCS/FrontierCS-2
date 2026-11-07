# Agent image of vllm_llm_serving_optimization: the shared Agent base plus the packages of the
# previous Agent image, the statement and the clean vLLM checkout (repo/). The H100 runner is
# docker/agent-gpu.Dockerfile. Composed with the Candidate overlay at Run time, so it must not
# depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates \
       curl git openssh-client python3 python3-pip python3-venv ripgrep tar \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization" \
      dev.frontier.vllm-commit="b8b302cde434df8c9289a2b465406b47ebab1c2d"

COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN chmod -R a+rX,a-w /frontier/task \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -perm /111 -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f ! -perm /111 -exec chmod 0444 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100
WORKDIR /workspace
