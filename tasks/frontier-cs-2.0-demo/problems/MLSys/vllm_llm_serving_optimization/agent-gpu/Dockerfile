# syntax=docker/dockerfile:1.7
# Preview build of the H100 Agent runner: the published runtime leaf (CUDA
# toolchain, vLLM wheel, clean source, Qwen3-Coder model, task runtime) plus
# the current trusted SSH bootstrap. Rendered by
# docs/transfers/vllm_llm_serving_optimization/tools/export_preview_tree.py.
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
FROM docker.io/joyemang33/fcs2-task-vllm_llm_serving_optimization-runtime:h100-r001@sha256:2b09056416a3a18049b2deeaf397f0c3afe5795dae1f4c57635ad206355bc89f
USER root
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime
RUN chown root:root /opt/frontier/task-runtime && chmod 0555 /opt/frontier/task-runtime
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap
ENTRYPOINT []
CMD []
WORKDIR /workspace
# The trusted bootstrap starts as root; task commands run as frontier (10001).
USER root
