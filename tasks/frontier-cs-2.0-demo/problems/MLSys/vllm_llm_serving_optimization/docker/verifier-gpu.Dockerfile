# syntax=docker/dockerfile:1.7
# Verifier GPU parent of vllm_llm_serving_optimization: the task's published runtime leaf plus
# the parent-side judge entry points and the shared SSH bootstrap (gpu-tools,
# tasks/shared/bases.json). The candidate vLLM server runs here; timing, tools and correctness
# run in the scheduler service and test sandboxes beside it.
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM docker.io/joyemang33/fcs2-task-vllm_llm_serving_optimization-runtime:h100-r001@sha256:2b09056416a3a18049b2deeaf397f0c3afe5795dae1f4c57635ad206355bc89f
USER root
RUN test -x /usr/bin/tar \
    && test -x /opt/frontier/runtime-python \
    && test -x /opt/frontier/task-runtime
COPY judge/variant_batch.py /opt/frontier/judge/variant_batch.py
# The relay forwards Verifier jobs to the scheduler service; it holds no key and its answers
# are verified on the CPU side.
COPY judge/scheduler_relay.py /opt/frontier/judge/scheduler_relay.py
COPY judge/scheduler_stream.py /opt/frontier/judge/scheduler_stream.py
RUN chown root:root /opt/frontier/judge/variant_batch.py /opt/frontier/judge/scheduler_relay.py /opt/frontier/judge/scheduler_stream.py \
    && chmod 0555 /opt/frontier/judge/variant_batch.py /opt/frontier/judge/scheduler_relay.py /opt/frontier/judge/scheduler_stream.py
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap
WORKDIR /workspace
ENTRYPOINT []
CMD []
# Driver starts the trusted bootstrap as root; all task SSH commands use frontier.
USER root
