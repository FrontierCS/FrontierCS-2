# syntax=docker/dockerfile:1.7
# Verifier for shockwave_dynamic_gpu_allocation: one CPU image, both modes. The platform wrapper
# replaces the image command and invokes /opt/frontier/verifier; task code has no
# Driver endpoint, provider credential, GPU capability, or cloud client.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM ubuntu:24.04

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir PyYAML==6.0.3 cvxpy==1.9.2 grpcio-tools==1.83.1 grpcio==1.83.1 gurobipy==13.0.3 matrix-completion==0.0.2 mosek==11.2.3 numpy==2.5.2 psutil==7.2.2 scikit-learn==1.9.0 scipy==1.18.1 zstandard


RUN install -d -o 65532 -g 65532 /frontier


# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/private_metrics.py /opt/frontier/private_metrics.py


# Source snapshots may carry restrictive modes. Grant read/traverse while
# preserving executable files so the immutable bundle works under the platform
# Verifier UID 65532 and read-only root filesystem.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task


USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
