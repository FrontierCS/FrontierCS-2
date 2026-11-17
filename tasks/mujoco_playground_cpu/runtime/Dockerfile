# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): CUDA
# 12.6.3 on Ubuntu 24.04, pinned by digest, with Python 3.12 and the source's hash-locked wheel
# closure (JAX 0.4.35 with its CUDA 12 wheels, Brax, MuJoCo MJX) in a virtual environment. It is
# built once for the agent's image and the verifier's, which build on it.
FROM nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e

USER root
ARG DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-gpu.lock /opt/frontier/requirements-gpu.lock
COPY runtime/setuptools.lock /opt/frontier/setuptools.lock
# The packages are installed from one dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that
# replaces the base's archive and NVIDIA's CUDA repository, which no snapshot covers; none of it
# comes from that repository. The compiler is for `gym` and `pytinyrenderer`, which publish no cp312
# wheel and are built from their locked source distributions; they build against the locked
# `setuptools`, installed first, in place of a build environment that pip would fetch with no hash
# (`--no-build-isolation`).
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh build-essential ca-certificates git libegl1 libgl1 libglib2.0-0 python3 python3-dev python3-venv tar \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && python3 -m venv /opt/frontier/venv \
    && /opt/frontier/venv/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --require-hashes --index-url https://pypi.org/simple \
       --requirement /opt/frontier/setuptools.lock \
    && /opt/frontier/venv/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --no-build-isolation --require-hashes --index-url https://pypi.org/simple \
       --requirement /opt/frontier/requirements-gpu.lock \
    && /opt/frontier/venv/bin/python -m pip check
ENV PATH=/opt/frontier/venv/bin:${PATH} MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
# Fails the build, not the first verification, if the locked stack cannot import or cannot
# serialise an exported function (CPU fallback: no GPU at build time).
RUN JAX_PLATFORMS=cpu python -c 'import jax, brax, mujoco, flax, optax, yaml, zstandard; from mujoco import mjx; from jax import export; from jax._src.export import serialization; assert (jax.__version__, brax.__version__, mujoco.__version__) == ("0.4.35", "0.12.1", "3.2.7")'
# A wrapper, not a symlink: a symlink to a virtual-environment interpreter outside its bin/
# directory loses the environment's site-packages.
RUN printf '#!/bin/sh\nexec /opt/frontier/venv/bin/python "$@"\n' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python
