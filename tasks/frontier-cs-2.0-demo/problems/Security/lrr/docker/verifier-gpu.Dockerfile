# syntax=docker/dockerfile:1.7
# GPU Verifier image of lrr: the fixed offline model runtime (llama.cpp server built for CUDA +
# Qwen3-8B Q4_K_M GGUF), ReDoSHunter under the Temurin 25 JRE, the judge Python closure, the
# pristine starter, this task's judge and the shared verifier modules. The CPU controller
# (docker/verifier.Dockerfile) brokers one Driver-owned A100 runner from this image; task code
# never receives a Modal, AWS or Kubernetes credential.
#
# llama.cpp (/app), the Temurin 25 JRE (/opt/java/openjdk) and the GGUF model are copied
# byte-for-byte from the previous release's Verifier GPU image by digest, so the judge's model
# server and detector stay bit-identical. Build context: this task's root, with .frontier-generated/.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS frontier_verifier
FROM ${GPU_TOOLS} AS frontier_gpu_tools
# nvidia/cuda:12.8.1-devel-ubuntu24.04, linux/amd64 manifest (the previous release's base).
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

USER root
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libgomp1 libnss-wrapper \
       openssh-client openssh-server python3 python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Judge Python closure in a venv, every artifact hash-pinned.
COPY docker/verifier-gpu.requirements-build.lock /opt/frontier/python-lock/requirements-build.lock
COPY docker/verifier-gpu.requirements.lock /opt/frontier/python-lock/requirements.lock
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

# Temurin JRE 25.0.4.1+1 for ReDoSHunter, as the previous release shipped it.
COPY --from=docker.io/joyemang33/fcs2-task-verifier-lrr-gpu@sha256:4fd4db49aebd09f65b253425d8327fbcdbd1df4b82e98f8e96af52ba6be109d5 /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk PATH="/opt/frontier/python/bin:/opt/java/openjdk/bin:${PATH}"
RUN java -version 2>&1 | grep -q '"25' && ln -s /opt/java/openjdk/bin/java /usr/local/bin/java

# llama.cpp b10868 (commit 304665fe, built for sm_80) and the fixed model weights
# (Qwen/Qwen3-8B-GGUF @7c41481f, Apache-2.0), as the previous release shipped them.
COPY --from=docker.io/joyemang33/fcs2-task-verifier-lrr-gpu@sha256:4fd4db49aebd09f65b253425d8327fbcdbd1df4b82e98f8e96af52ba6be109d5 /app/ /app/
COPY --from=docker.io/joyemang33/fcs2-task-verifier-lrr-gpu@sha256:4fd4db49aebd09f65b253425d8327fbcdbd1df4b82e98f8e96af52ba6be109d5 /models/Qwen3-8B-Q4_K_M.gguf /models/Qwen3-8B-Q4_K_M.gguf
# libcuda.so.1 comes from the GPU host driver at run time, so the server cannot start during
# the build; check that every other shared library resolves.
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

# The judge side of the task; pristine/ is the starter the shared verifier applies the patch to.
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
# The shared verifier (edit_scope, then evaluate(tree)), from the shared Verifier base v10 by digest.
COPY --from=frontier_verifier /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/edit_scope.py /opt/frontier/
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml

# The judge runs the submitted prompt constructor with the shared isolate launcher (userns
# backend on the runner): own namespaces, no judge files, no network, no GPU.
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/isolate.py
# /opt/frontier on the judge interpreter's default import path (frontier.pth), so the judge and
# every worker process it starts import isolate by name.
RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier

# Grant read/traverse to both platform UIDs while keeping the bundle immutable.
RUN /opt/frontier/runtime-python -c 'import zstandard, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/*.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap

USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_COUNT=1
WORKDIR /frontier
