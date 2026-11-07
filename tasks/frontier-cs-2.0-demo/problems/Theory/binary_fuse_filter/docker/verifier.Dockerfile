# Verifier image of binary_fuse_filter, both modes: the shared Verifier base plus this task's
# compiler, its judge (evaluator, measurement code, trusted C harness and bridge, exact
# reference headers and reference-only calibration) and the starter repo the
# shared verifier overlays the submission on. Build context: this task's root, with
# .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier binary_fuse_filter" \
      dev.frontier.problem-id="binary_fuse_filter"

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
