# Verifier image of cute_crest, both modes: the shared Verifier base plus this task's packages,
# judge bundle, generated config, and the harness and instrumented programs built once here.
# The platform wrapper invokes /opt/frontier/verifier; task evaluator code never receives a
# Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils bsdutils build-essential ca-certificates coreutils dpkg git libncurses-dev make patch perl python3 python3-pip time \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier cute_crest" \
      dev.frontier.problem-id="cute_crest"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml

# Build the harness and the instrumented programs once, outside the task bundle (a
# verification copies the bundle, not this). Program sources are fetched by pinned URL and
# SHA-256 and stay in the image, with their licence files, beside the binaries built from
# them (/opt/crest-task/src).
RUN python3 /opt/frontier/task/build_benchmarks.py /opt/crest-task

RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
