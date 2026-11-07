# Verifier image of runlts_predictor, both modes: the shared Verifier base plus this task's packages
# (C++ toolchain, zlib, prlimit, libclang for the source audit), the 105 private CBP traces from
# their digest-pinned image, the judge bundle and generated config. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip util-linux zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 libclang==18.1.1

# The private traces (about 15 GB), byte for byte and with the modes the qualified release had
# (a+rX, a-w); a recursive chmod here would copy the whole layer.
COPY --from=docker.io/joyemang33/fcs2-task-verifier-runlts_predictor@sha256:74b8ba63764bae367ee21dd4d688ddc7b9dfac66e64a7735c920c03e2b825ff7 /opt/runlts-private-data /opt/runlts-private-data

LABEL org.opencontainers.image.title="FrontierCS2 verifier runlts_predictor" \
      dev.frontier.problem-id="runlts_predictor"

COPY judge /opt/frontier/task
# evaluator.py reads ROOT/private_data/traces.
RUN ln -s /opt/runlts-private-data /opt/frontier/task/private_data
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0555 /opt/runlts-private-data

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
