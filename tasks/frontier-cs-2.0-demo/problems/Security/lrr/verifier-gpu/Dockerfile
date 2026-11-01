# syntax=docker/dockerfile:1.7
# Verifier GPU image for lrr: the fixed offline model runtime (llama.cpp
# server built for CUDA + Qwen3-8B Q4_K_M GGUF), ReDoSHunter under OpenJDK 25,
# the judge Python closure, the pristine public tree and the judge (frozen
# workloads, reference patch, beat spec). The CPU controller
# (verifier/Dockerfile) brokers one Driver-owned A100 runner from this image;
# task code never receives a Modal, AWS or Kubernetes credential.
#
# Build context is this task's own root directory:
#   docker build -f verifier-gpu/Dockerfile -t <tag> .
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE} AS frontier_gpu_ssh

# eclipse-temurin 25 JDK (Docker Hub library image), the author's qualified Java 25.
FROM eclipse-temurin@sha256:dbb43d81099eec91e0800226ed2296d03146ee7857165b95112372ad0d67e87f AS java25

# llama.cpp b10868 (commit 304665fe7ac957df95e3ff8c8c4ffdf92dd6ffa3), the same
# revision as the author's ghcr.io server-cuda runtime, built here from source
# for the A100 (sm_80). nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 manifest
# (the same base Fast KV uses).
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS llama_build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git init /src/llama.cpp \
    && git -C /src/llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp.git 304665fe7ac957df95e3ff8c8c4ffdf92dd6ffa3 \
    && git -C /src/llama.cpp checkout --quiet FETCH_HEAD \
    && test "$(git -C /src/llama.cpp rev-parse HEAD)" = 304665fe7ac957df95e3ff8c8c4ffdf92dd6ffa3
# Same CMake configuration as upstream .devops/cuda.Dockerfile (dynamic CPU
# backend variants, CUDA backend), with the CUDA architecture limited to sm_80.
RUN cd /src/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release --target llama-server -j"$(nproc)" \
    && install -d /app \
    && find build -name '*.so*' -exec cp -P {} /app \; \
    && install -m 0555 build/bin/llama-server /app/llama-server

FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libgomp1 libnss-wrapper \
       openssh-client openssh-server python3 python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Judge Python closure in a venv, every artifact hash-pinned.
COPY verifier-gpu/requirements-build.lock /opt/frontier/python-lock/requirements-build.lock
COPY verifier-gpu/requirements.lock /opt/frontier/python-lock/requirements.lock
RUN /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --require-hashes --index-url https://pypi.org/simple \
       -r /opt/frontier/python-lock/requirements-build.lock \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --require-hashes --no-build-isolation --index-url https://pypi.org/simple \
       -r /opt/frontier/python-lock/requirements.lock \
    && /opt/frontier/python/bin/python -c 'import yaml, editdistance, exrex, regex, timeout_decorator, tqdm, xeger, zstandard'
ENV PATH="/opt/frontier/python/bin:${PATH}"
# runtime-python is a wrapper, not a symlink, so the venv stays active over SSH.
RUN printf '%s\n' '#!/bin/sh' 'export PATH="/opt/frontier/python/bin:/opt/java/openjdk/bin:$PATH"' \
      'exec /opt/frontier/python/bin/python3 "$@"' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python

# OpenJDK 25 for ReDoSHunter.
COPY --from=java25 /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk PATH="/opt/frontier/python/bin:/opt/java/openjdk/bin:${PATH}"
RUN java -version 2>&1 | grep -q '"25' && ln -s /opt/java/openjdk/bin/java /usr/local/bin/java

# Fixed model server and weights (Qwen/Qwen3-8B-GGUF @7c41481f, Apache-2.0).
COPY --from=llama_build /app/ /app/
ADD --checksum=sha256:d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785 \
    https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf \
    /models/Qwen3-8B-Q4_K_M.gguf
# libcuda.so.1 comes from the GPU host driver at run time, so the server cannot
# start during the build; check that every other shared library resolves.
RUN chmod 0555 /app /models && chmod 0444 /models/Qwen3-8B-Q4_K_M.gguf \
    && test -x /app/llama-server \
    && test -f /app/libggml-cuda.so \
    && ! for f in /app/llama-server /app/*.so; do LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64 ldd "$f"; done | grep 'not found' | grep -v 'libcuda\.so'
ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN install -d -o 65532 -g 65532 /frontier
RUN if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; else useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier; fi \
    && usermod --shell /bin/bash --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace

LABEL org.opencontainers.image.title="FrontierCS2 verifier lrr" \
      dev.frontier.problem-id="lrr" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes (minus its README), because the evaluator applies the submitted
# patch against a pristine copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py

# Grant read/traverse to both platform UIDs while keeping the bundle immutable.
# The submitted prompt policy runs in judge/policy_sandbox.py: unprivileged user,
# mount, PID, network and IPC namespaces with the judge hidden.
RUN /opt/frontier/runtime-python -c 'import zstandard, yaml' \
    && test -x /usr/bin/unshare \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && find /opt/frontier -maxdepth 1 -type f -name '*.py' -exec chmod 0444 {} + \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/harbor/app -type f -name '*.sh' -exec chmod 0555 {} +
COPY --from=frontier_gpu_ssh /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_SECONDS=3600 FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
