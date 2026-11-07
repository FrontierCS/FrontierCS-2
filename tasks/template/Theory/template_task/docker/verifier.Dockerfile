# Verifier image of template_task (train and eval): the shared Verifier base plus this task's
# packages, judge and generated config.
#
# Rules:
#   - the final stage is FROM ${VERIFIER_BASE} (ARG VERIFIER_BASE, no default): the build
#     injects tasks/shared/bases.json["verifier"], the one current base;
#   - the judge lives at /opt/frontier/task; the shared entry /opt/frontier/verifier applies the
#     submission with the shared edit_scope (rules in the generated config.yaml) and calls
#     evaluate(tree) from a writable copy of /opt/frontier/task/evaluator.py;
#   - the judge never receives a Modal, AWS or Kubernetes credential and never reaches the
#     network: it runs with no egress;
#   - the starter is repo/ itself, copied to /opt/frontier/task/pristine (one copy, no
#     judge/harbor/app duplicate);
#   - pin Python packages the judge imports (the base pins zstandard).
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier template_task" \
      dev.frontier.problem-id="template_task"

COPY judge /opt/frontier/task
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
