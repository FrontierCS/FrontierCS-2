# Verifier image of clusterwise_spgemm, both modes: the shared Verifier base plus this task's
# packages, judge bundle and generated config. The selected dataset tar is supplied at run time
# (FRONTIER_DATASET_PATH) and extracted by judge/dataset_loader.py; no dataset body is in the
# image. The platform wrapper invokes /opt/frontier/verifier; task evaluator code never receives
# a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libtbb-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir numpy==2.3.4 scipy==1.16.3

LABEL org.opencontainers.image.title="FrontierCS2 verifier clusterwise_spgemm" \
      dev.frontier.problem-id="clusterwise_spgemm"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to and builds its baseline kernel object from); its source is repo/ itself.
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
