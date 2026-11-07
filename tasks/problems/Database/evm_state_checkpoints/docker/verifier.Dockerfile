# Verifier image of evm_state_checkpoints, both modes: the shared Verifier base plus this
# task's packages, judge bundle and generated config. The platform wrapper invokes
# /opt/frontier/verifier; task evaluator code never receives a Modal, AWS or Kubernetes
# credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
# The package set of buildpack-deps:noble, which the task's images were built on before
#, plus cmake: the same compilers, build tools and -dev libraries.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       autoconf automake bzip2 ca-certificates cmake curl default-libmysqlclient-dev dpkg-dev file g++ gcc git gnupg imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make mercurial netbase openssh-client patch procps python3 sq subversion unzip wget xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier evm_state_checkpoints" \
      dev.frontier.problem-id="evm_state_checkpoints"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge builds its
# shipped and reference backends from); its source is repo/ itself, not a second copy.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task

# Spectrum's CMake project fetches pinned dependencies and caches absolute paths; the Verifier
# has no egress. Build the cache at image-build time at the path the run-scoped writable mount
# provides, publish it as a read-only seed, and the evaluator copies it back at run time.
RUN FRONTIER_SPECTRUM_CACHE=/frontier/spectrum-cache PYTHONPATH=/opt/frontier/task:/opt/frontier python3 -c 'import evaluator; r = evaluator.prepare(); assert r.get("ready"), r' \
    && chmod 0755 /opt/frontier \
    && cp -a /frontier/spectrum-cache /opt/frontier/spectrum-cache-seed \
    && chmod 0555 /opt/frontier \
    && rm -rf /frontier/spectrum-cache \
    && chmod -R a+rX /opt/frontier/spectrum-cache-seed

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_SPECTRUM_CACHE=/frontier/spectrum-cache FRONTIER_SPECTRUM_CACHE_SEED=/opt/frontier/spectrum-cache-seed
WORKDIR /frontier
