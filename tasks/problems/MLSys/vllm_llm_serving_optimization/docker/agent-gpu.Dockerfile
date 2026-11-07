# syntax=docker/dockerfile:1.7
# H100 Agent runner of vllm_llm_serving_optimization: the task's published runtime leaf (CUDA
# toolchain, vLLM 0.11.0 wheel, clean source at /opt/vllm-clean, Qwen3-Coder model at
# /opt/frontier/model) plus the task runtime and the shared SSH bootstrap (gpu-tools,
# tasks/shared/bases.json).
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM docker.io/joyemang33/fcs2-task-vllm_llm_serving_optimization-runtime:h100-r001@sha256:2b09056416a3a18049b2deeaf397f0c3afe5795dae1f4c57635ad206355bc89f
USER root
COPY gpu/task_runtime.py /opt/frontier/task-runtime
RUN chown root:root /opt/frontier/task-runtime && chmod 0555 /opt/frontier/task-runtime
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap
ENTRYPOINT []
CMD []
WORKDIR /workspace
# The trusted bootstrap starts as root; task commands run as frontier (10001).
USER root
