# Verifier image of edlib_exact_alignment, both modes: the shared Verifier base (task.toml
# `base`) plus this task's packages, judge bundle and generated config. The platform wrapper
# invokes /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or
# Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git python3-yaml \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir pytest==9.1.1 pluggy==1.6.0

LABEL org.opencontainers.image.title="FrontierCS2 verifier edlib_exact_alignment" \
      dev.frontier.problem-id="edlib_exact_alignment"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
