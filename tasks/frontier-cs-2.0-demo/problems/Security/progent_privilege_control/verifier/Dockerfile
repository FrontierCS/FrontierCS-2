# syntax=docker/dockerfile:1.7
# Verifier for progent_privilege_control: one image, both modes. The platform
# wrapper invokes /opt/frontier/verifier. The episodes reach the task model only
# through the run's additional_keys inventory (Driver proxy); the evaluator holds
# no other credential.
#
# Build context is this task's own root directory:
#   docker build -f verifier/Dockerfile -t <tag> .
FROM ubuntu:24.04

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libseccomp2 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY agent-base/frontier/task/requirements-lock.txt /opt/progent/requirements-lock.txt
RUN python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --no-deps -r /opt/progent/requirements-lock.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain, zstandard"

RUN install -d -o 65532 -g 65532 /frontier \
    && if ! getent passwd 65532 >/dev/null; then useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier; fi
ENV HOME=/frontier

LABEL org.opencontainers.image.title="FrontierCS2 verifier progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control" \
      dev.frontier.generator="frontiersmith2"

# The judge side. `harbor/app` is the same tree the Agent image publishes as
# /frontier/task/repo; the evaluator applies the submitted patch to a copy of it
# and runs the pristine tree as the reference.
COPY judge /opt/frontier/task
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py

RUN printf '#!/bin/sh\nexec /opt/progent/venv/bin/python "$@"\n' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0555 /opt/frontier \
    && find /opt/frontier -maxdepth 1 -type f -name '*.py' -exec chmod 0444 {} + \
    && rm -rf /opt/frontier/task/__pycache__ /opt/frontier/task/trusted/secagent/__pycache__ \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/harbor/app -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
