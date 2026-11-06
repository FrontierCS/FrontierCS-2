# Verifier image of diffusion_policy_cpu, both modes: the shared Verifier base plus this task's
# measurement runtime, judge bundle and generated config. The measurement runtime is the
# release's: the CPython 3.10.20 interpreter of the pinned python image (copied by digest to
# /opt/python3.10) and, built from it by judge/prepare_runtime.py, the venv
# /opt/diffusion-policy/python with the hash-pinned wheels and the official checkpoint. The
# shared verifier itself runs on the base's Python. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates git libgl1 libglib2.0-0 \
       libbz2-1.0 libcrypt1 libdb5.3t64 libexpat1 libffi8 libgdbm6t64 libgssapi-krb5-2 liblzma5 \
       libncursesw6 libnsl2 libreadline8t64 libsqlite3-0 libssl3t64 libtirpc3t64 libuuid1 zlib1g \
       python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3

# The release's interpreter, byte for byte (python:3.10.20-slim-bookworm), relocated.
COPY --from=docker.io/library/python@sha256:6dafe336b64dbf76f7ec872d07e58b5c7de0e2ea5d4c0853a05c3b6be3278b00 /usr/local /opt/python3.10
RUN echo /opt/python3.10/lib > /etc/ld.so.conf.d/python3.10.conf && ldconfig \
    && /opt/python3.10/bin/python3.10 -c 'import ssl, sqlite3, ctypes, lzma, bz2, readline, uuid; print("python3.10 ok")' \
    && /opt/python3.10/bin/python3.10 -m pip install --no-cache-dir PyYAML==6.0.3 zstandard==0.25.0

LABEL org.opencontainers.image.title="FrontierCS2 verifier diffusion_policy_cpu" \
      dev.frontier.problem-id="diffusion_policy_cpu"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && PYTHONDONTWRITEBYTECODE=1 /opt/python3.10/bin/python3.10 /opt/frontier/task/prepare_runtime.py \
    && chmod 0555 /opt/frontier

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
