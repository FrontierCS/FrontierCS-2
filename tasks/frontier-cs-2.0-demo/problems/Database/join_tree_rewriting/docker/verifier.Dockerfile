# Verifier image of join_tree_rewriting, both modes: the shared Verifier base plus this task's
# judge bundle (evaluator, reproduced join executor, Parquet data, workloads) and generated
# config. The platform wrapper invokes /opt/frontier/verifier; task evaluator code never
# receives a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
LABEL org.opencontainers.image.title="FrontierCS2 verifier join_tree_rewriting" \
      dev.frontier.problem-id="join_tree_rewriting"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} + \
    && /opt/frontier/task/judge/bin/join_runner --help | grep -q -- --validation-outfile

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
