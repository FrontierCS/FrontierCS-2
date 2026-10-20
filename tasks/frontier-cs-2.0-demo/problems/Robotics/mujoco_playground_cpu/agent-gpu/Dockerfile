# syntax=docker/dockerfile:1.7
# GPU runner image for mujoco_playground_cpu. The platform creates one of these on the
# solver's request and connects it through its own SSH gateway; the solver never
# holds a provider credential and never learns this instance's address.
#
# It carries the toolchain and the read-only source tree, and no part of either
# suite: what runs here is the solver's own training and the repository's own
# public code, and a runner that could see the graded workload would be a
# second grading path.
# The publisher supplies an immutable image built from FrontierCS2's trusted
# `driver/images/gpu-ssh-base/Dockerfile` recipe.
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh
FROM nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e

# Ubuntu 24.04 ships CPython 3.12 as its system interpreter, so the pinned base
# image fixes the Python minor version without any third-party apt source. The
# scientific stack is a hash-locked wheel closure installed into a root-owned
# virtual environment: jax[cuda12]==0.4.35 brings its own CUDA 12 runtime wheels
# (nvidia-cuda-nvcc-cu12 is held below 12.7; newer nvcc wheels break `import jax`
# 0.4.35). gym and pytinyrenderer publish no cp312 wheel and are built from
# their hash-locked source distributions, hence build-essential/python3-dev.
USER root
ARG DEBIAN_FRONTEND=noninteractive
COPY agent-base/frontier/task/runtime/requirements-gpu.lock /opt/frontier/requirements-gpu.lock
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libegl1 libgl1 libglib2.0-0 libnss-wrapper openssh-client openssh-server python3 python3-dev python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && python3 -m venv /opt/frontier/venv \
    && /opt/frontier/venv/bin/python -m pip install --no-cache-dir --require-hashes -r /opt/frontier/requirements-gpu.lock \
    && chmod -R a+rX,go-w /opt/frontier/venv
ENV PATH=/opt/frontier/venv/bin:${PATH} MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
# Fail the build, not the first verification, if the locked stack cannot import
# or cannot serialise an exported function (CPU fallback: no GPU at build time).
RUN JAX_PLATFORMS=cpu python -c 'import jax, brax, mujoco, flax, optax, yaml, zstandard; from mujoco import mjx; from jax import export; from jax._src.export import serialization; assert (jax.__version__, brax.__version__, mujoco.__version__) == ("0.4.35", "0.12.1", "3.2.7")'

RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Keep the immutable version label from invalidating reusable dependency
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner mujoco_playground_cpu" \
      dev.frontier.problem-id="mujoco_playground_cpu" \
      dev.frontier.generator="manual-authoring"

# The public tree already contains the pinned Unitree Go1 MuJoCo Menagerie
# assets under mujoco_playground/external_deps, so importing mujoco_playground
# never needs the network.
COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime

# Snapshot only public toolchain settings, never the build environment's
# credentials. SSH has a deliberately minimal environment.
# /opt/frontier/runtime-python is a wrapper, not a symlink: a symlink to a
# virtual-environment interpreter outside its bin/ directory loses the
# environment's site-packages.
RUN printf '#!/bin/sh\nexec /opt/frontier/venv/bin/python "$@"\n' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} +

COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
