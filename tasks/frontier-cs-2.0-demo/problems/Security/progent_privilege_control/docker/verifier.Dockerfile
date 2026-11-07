# Verifier image of progent_privilege_control, both modes: the shared Verifier base plus this
# task's Python environment (the same pinned closure as the Agent), judge bundle (evaluator,
# trusted harness, policy worker, the full upstream tree with the held-out user tasks) and
# generated config. The shared verifier runs on the task's environment (runtime-python below),
# because the evaluator runs the AgentDojo agent in-process. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly; its only model access is the run's key.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY repo/tools/requirements-lock.txt /opt/progent/requirements-lock.txt
RUN python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --no-deps -r /opt/progent/requirements-lock.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain, zstandard" \
    && printf '#!/bin/sh\nexec /opt/progent/venv/bin/python "$@"\n' > /opt/frontier/runtime-python.task \
    && chmod 0555 /opt/frontier/runtime-python.task \
    && mv /opt/frontier/runtime-python.task /opt/frontier/runtime-python

LABEL org.opencontainers.image.title="FrontierCS2 verifier progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on; its source is repo/ itself.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && rm -rf /opt/frontier/task/__pycache__ /opt/frontier/task/trusted/secagent/__pycache__ \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
