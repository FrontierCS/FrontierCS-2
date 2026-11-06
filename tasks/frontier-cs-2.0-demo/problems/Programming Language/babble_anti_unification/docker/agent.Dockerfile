# Agent image of babble_anti_unification: the shared Agent base plus this task's packages, the
# Rust 1.64 toolchain the judge builds with, the statement, the starter repo and the generated
# submission policy. Composed with the Candidate overlay at Run time, so it must not depend on
# its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

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

LABEL org.opencontainers.image.title="FrontierCS2 task babble_anti_unification" \
      dev.frontier.problem-id="babble_anti_unification"

# The solver-visible tree, and nothing else: the evaluator, reference, workloads and the
# beat-reference specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
