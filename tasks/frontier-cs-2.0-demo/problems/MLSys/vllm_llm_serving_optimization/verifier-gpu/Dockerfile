# syntax=docker/dockerfile:1.7
# Preview build of the Verifier GPU parent: the published runtime leaf plus the
# parent-side judge entry points and the current trusted SSH bootstrap.
# Rendered by docs/transfers/vllm_llm_serving_optimization/tools/export_preview_tree.py.
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
FROM docker.io/joyemang33/fcs2-task-vllm_llm_serving_optimization-runtime:h100-r001@sha256:2b09056416a3a18049b2deeaf397f0c3afe5795dae1f4c57635ad206355bc89f
USER root
RUN test -x /usr/bin/tar \
    && test -x /opt/frontier/runtime-python \
    && test -x /opt/frontier/task-runtime
COPY judge/variant_batch.py /opt/frontier/judge/variant_batch.py
# The relay forwards Verifier jobs to the declared scheduler sidecar; it holds
# no key and its answers are verified on the CPU side.
COPY judge/scheduler_relay.py /opt/frontier/judge/scheduler_relay.py
COPY judge/scheduler_stream.py /opt/frontier/judge/scheduler_stream.py
RUN chown root:root /opt/frontier/judge/variant_batch.py /opt/frontier/judge/scheduler_relay.py /opt/frontier/judge/scheduler_stream.py \
    && chmod 0555 /opt/frontier/judge/variant_batch.py /opt/frontier/judge/scheduler_relay.py /opt/frontier/judge/scheduler_stream.py
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap
WORKDIR /workspace
ENTRYPOINT []
CMD []
# Driver starts the trusted bootstrap as root; all task SSH commands use frontier.
USER root
