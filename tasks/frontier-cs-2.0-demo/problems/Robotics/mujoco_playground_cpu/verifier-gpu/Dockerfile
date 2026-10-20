# syntax=docker/dockerfile:1.7
# Verifier for mujoco_playground_cpu: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier-gpu/Dockerfile -t <tag> .
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

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

# Versioned metadata belongs after the reusable toolchain layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier mujoco_playground_cpu" \
      dev.frontier.problem-id="mujoco_playground_cpu" \
      dev.frontier.generator="manual-authoring"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes: the pristine MuJoCo Playground checkout (with the pinned Unitree
# Go1 MuJoCo Menagerie assets already inside it, so nothing is cloned at run
# time) plus the default frontier_solution package. The evaluator applies the
# submitted patch to one copy and trains/evaluates the reference from another.
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py

# Source snapshots may be created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user. Grant read/traverse while preserving executable files; the bundle and
# the Python runtime stay root-owned and immutable to that user, which the
# trusted runner requires (judge_deps/runtime.py).
# /opt/frontier/runtime-python is a wrapper, not a symlink: a symlink to a
# virtual-environment interpreter outside its bin/ directory loses the
# environment's site-packages.
RUN printf '#!/bin/sh\nexec /opt/frontier/venv/bin/python "$@"\n' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && chmod 0444 /opt/frontier/*.py \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && /opt/frontier/runtime-python -B -c 'import json, sys; sys.path.insert(0, "/opt/frontier/task/judge_deps"); import runtime; runtime.admit_pristine("/opt/frontier/task/harbor/app", runtime.load_source_manifest()); runtime.admit_trusted_children(runtime.load_source_manifest())'
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=H100 FRONTIER_VERIFIER_GPU_SECONDS=5400 FRONTIER_VERIFIER_GPU_EVAL_SECONDS=14400 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
