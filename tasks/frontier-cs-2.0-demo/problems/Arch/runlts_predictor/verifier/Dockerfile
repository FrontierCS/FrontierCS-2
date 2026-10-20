# syntax=docker/dockerfile:1.7
# Verifier for runlts_predictor: one image, both modes. The platform wrapper replaces
# the image command and invokes /opt/frontier/verifier. The shared adapter may use the
# verification-scoped Driver GPU capability; task evaluator code never receives
# a Modal, AWS, or Kubernetes credential and must not reach S3 directly.
#
# Build context is this task's own root directory, same as agent-base/Dockerfile:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM docker.io/joyemang33/fcs2-task-verifier-runlts_predictor@sha256:74b8ba63764bae367ee21dd4d688ddc7b9dfac66e64a7735c920c03e2b825ff7 AS pinned_traces

FROM ubuntu:24.04

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip util-linux zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 libclang==18.1.1 zstandard


RUN install -d -o 65532 -g 65532 /frontier
# Jupyter and other NSS clients need an account for the numeric Verifier UID.
RUN if ! getent passwd 65532 >/dev/null; then useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier; fi
ENV HOME=/frontier

# Versioned metadata belongs after the reusable toolchain and model layers.
LABEL org.opencontainers.image.title="FrontierCS2 verifier runlts_predictor" \
      dev.frontier.problem-id="runlts_predictor" \
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
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path('"'"'/opt/frontier/runtime-python'"'"'); output = path.open('"'"'x'"'"'); output.write('"'"'#!/bin/sh\nexec '"'"' + shlex.quote(sys.executable) + '"'"' "$@"\n'"'"'); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && find /opt/frontier -maxdepth 1 -type f -name '*.py' -exec chmod 0444 {} + \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/harbor/app -type f -name '*.sh' -exec chmod 0555 {} +


# Task-declared judge build steps (runtime.judge_build_commands).
# Reuse only the immutable trace bytes; no old judge/runtime layers reach the final image.
COPY --from=pinned_traces /opt/runlts-private-data /opt/runlts-private-data
RUN chmod 0555 /opt/runlts-private-data \
    && ln -s /opt/runlts-private-data /opt/frontier/task/private_data

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
