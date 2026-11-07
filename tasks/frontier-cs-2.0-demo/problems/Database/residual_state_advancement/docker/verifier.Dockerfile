# Verifier image of residual_state_advancement, both modes: the shared Verifier base plus this
# task's packages, judge bundle and generated config. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
# The Python packages of the task's previous image, pinned to the versions it had.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --no-deps cloudpickle==3.1.2 duckdb==1.5.1 joblib==1.6.0 numpy==2.5.4 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2026.5 PyYAML==6.0.2 scikit-learn==1.7.2 scipy==1.18.1 six==1.17.0 threadpoolctl==3.7.0 tzdata==2026.5

LABEL org.opencontainers.image.title="FrontierCS2 verifier residual_state_advancement" \
      dev.frontier.problem-id="residual_state_advancement"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference and baseline patches to); its source is repo/ itself, not a second copy.
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
