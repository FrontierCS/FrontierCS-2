# syntax=docker/dockerfile:1.7
# Verifier control image of nano_world_model: the shared Verifier base, the shared GPU adapter
# (gpu-tools), this task's judge and starter, and the suite's public data. The judge runs here;
# the candidate's sampling goes to a [test_sandboxes.candidate] runner (Agent GPU image), the
# reference and the decode to a [test_sandboxes.trusted] runner (docker/verifier-gpu.Dockerfile),
# both through run_sandboxes() (task.toml). CPU PyTorch only builds the public data
# (judge/public_inputs.py: per clip the context frame, no scored frame).
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnss-wrapper openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --no-deps \
    'torch @ https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=8800deef0026011d502c0c256cc4b67d002347f63c3a38cd8e45f1f445c61364' \
    filelock==3.15.4 fsspec==2024.6.1 jinja2==3.1.4 MarkupSafe==2.1.5 mpmath==1.3.0 \
    networkx==3.3 sympy==1.13.3 typing-extensions==4.12.2

LABEL org.opencontainers.image.title="FrontierCS2 verifier control nano_world_model" \
      dev.frontier.problem-id="nano_world_model"

# The judge and the starter (the shared verifier overlays the submission's allowed files).
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
RUN python3 /opt/frontier/task/public_inputs.py /opt/frontier/task/judge_data /opt/frontier/task/public_data
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
