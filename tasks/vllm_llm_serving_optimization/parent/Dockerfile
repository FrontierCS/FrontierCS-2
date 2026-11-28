# The judges' H100 parent: the task's runtime with the parent-side entry point, which prepares the
# submitted vLLM variant and starts and stops its server. The candidate's server runs here, as
# uid 1000, which owns /workspace; timing, tools and correctness run in the scheduler and the SWE
# containers of the parent's pod.
FROM runtime
USER root
COPY --chmod=0555 judge/variant_batch.py /opt/frontier/judge/variant_batch.py
RUN test -x /usr/bin/tar && test -x /opt/frontier/runtime-python && test -x /opt/frontier/task-runtime \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 verifier parent vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization"

USER 1000:1000
WORKDIR /workspace
