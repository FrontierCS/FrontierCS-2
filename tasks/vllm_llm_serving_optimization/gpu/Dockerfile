# The agent's H100 runner: the task's runtime, with the agent's user. The source's runner is the
# same runtime behind its SSH bootstrap, which made the `frontier` account there; a Pumpkin runner
# runs its holder's commands as uid 1000, user `ubuntu` as in the shared agent base, which owns
# /workspace, where the launch helper prepares each vLLM variant.
FROM runtime
USER root
RUN test -x /opt/frontier/runtime-python && test -x /opt/frontier/task-runtime \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization"

USER 1000:1000
ENV FRONTIER_GPU_TYPE=H100
WORKDIR /workspace
