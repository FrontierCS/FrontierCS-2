# Verifier image of shockwave_dynamic_gpu_allocation, both modes: the shared Verifier base plus
# this task's packages (pinned as before), judge bundle (evaluator, private traces, reference)
# and generated config. The platform wrapper invokes /opt/frontier/verifier; task evaluator code
# never receives a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir PyYAML==6.0.3 cvxpy==1.9.2 grpcio-tools==1.83.1 grpcio==1.83.1 gurobipy==13.0.3 matrix-completion==0.0.2 mosek==11.2.3 numpy==2.5.2 psutil==7.2.2 scikit-learn==1.9.0 scipy==1.18.1

LABEL org.opencontainers.image.title="FrontierCS2 verifier shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation"

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
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
