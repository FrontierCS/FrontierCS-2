# Verifier image of babble_anti_unification, both modes: the shared Verifier base plus this
# task's packages, the Rust 1.64 toolchain it builds submissions with, its judge bundle and
# generated config. The platform wrapper invokes /opt/frontier/verifier; task evaluator code
# never receives a Modal, AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Rust 1.64.0 installed the way the rust:1.64 image (used before) installs it: rustup 1.25.1,
# minimal profile, under /usr/local; the rustup-init digest is that image's rustup binary.
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo \
    RUSTUP_TOOLCHAIN=1.64.0 PATH=/usr/local/cargo/bin:${PATH}
RUN python3 -c 'import hashlib, sys, urllib.request; body = urllib.request.urlopen("https://static.rust-lang.org/rustup/archive/1.25.1/x86_64-unknown-linux-gnu/rustup-init").read(); assert hashlib.sha256(body).hexdigest() == "5cc9ffd1026e82e7fb2eec2121ad71f4b0f044e88bca39207b3f6b769aaa799c"; open("/tmp/rustup-init", "wb").write(body)' \
    && chmod 0755 /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.64.0 --default-host x86_64-unknown-linux-gnu \
    && rm /tmp/rustup-init \
    && chmod -R a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

LABEL org.opencontainers.image.title="FrontierCS2 verifier babble_anti_unification" \
      dev.frontier.problem-id="babble_anti_unification"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on and the reference patch applies
# to (with its vendored Cargo dependencies); its source is repo/ itself.
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
