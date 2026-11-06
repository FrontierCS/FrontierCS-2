# Verifier image of persistent_treap_update_engine, both modes: the shared Verifier base plus
# this task's packages, judge bundle and generated config. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libmimalloc-dev python3 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier persistent_treap_update_engine" \
      dev.frontier.problem-id="persistent_treap_update_engine"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on, the reference patch applies to,
# and the fixed serial scaffold is built from; its source is repo/ itself.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
