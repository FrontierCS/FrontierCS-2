# Verifier image of lwe_structured_recovery, both modes: the shared Verifier base plus this
# task's judge and the starter repo (the checker and catalog the judge trusts are read from
# pristine/, never from the submission). Build context: this task's root, with
# .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
LABEL org.opencontainers.image.title="FrontierCS2 verifier lwe_structured_recovery" \
      dev.frontier.problem-id="lwe_structured_recovery"

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
