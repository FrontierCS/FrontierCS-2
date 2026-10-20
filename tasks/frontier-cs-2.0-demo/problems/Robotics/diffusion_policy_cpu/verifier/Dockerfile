# syntax=docker/dockerfile:1.7
# Verifier for diffusion_policy_cpu: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM python@sha256:6dafe336b64dbf76f7ec872d07e58b5c7de0e2ea5d4c0853a05c3b6be3278b00

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates git libgl1 libglib2.0-0 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 zstandard zstandard==0.25.0


RUN install -d -o 65532 -g 65532 /frontier
# Jupyter and other NSS clients need an account for the numeric Verifier UID.
RUN if ! getent passwd 65532 >/dev/null; then useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier; fi
ENV HOME=/frontier

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier diffusion_policy_cpu" \
      dev.frontier.problem-id="diffusion_policy_cpu" \
      dev.frontier.generator="frontiersmith2"

# The judge side of the task. `harbor/app` is the same tree the Agent image
# publishes, because the evaluator applies the submitted patch against a pristine
# copy of it; everything beside it is judge-only.
COPY judge /opt/frontier/task
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py


# Source snapshots are created under umask 0077 and Docker preserves those
# modes on COPY. The Modal SSH daemon evaluates as the unprivileged `frontier`
# user, so merely removing write bits turns 0600 files into 0400 root-only
# files. Grant read/traverse while preserving executable files; the bundle
# stays immutable and becomes usable by both platform UIDs. Match the Agent's
# shell-entrypoint modes even when a sealed build context removed execute bits.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/harbor/app -type f -name '*.sh' -exec chmod 0555 {} +


# Task-declared judge build steps (runtime.judge_build_commands).
RUN python3 /opt/frontier/task/prepare_runtime.py

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
