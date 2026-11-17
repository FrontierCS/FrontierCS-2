# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3
# Rust 1.94.1 as the source installs it, rustup 1.29.0 and the minimal profile under /usr/local, but
# with the rustup-init file and the installed toolchain each pinned by a digest (`fetch/rustup.py`).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUSTUP_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu RUST_VERSION=1.94.1 PATH=/usr/local/cargo/bin:${PATH}
RUN --mount=type=bind,source=fetch/rustup.py,target=/context/fetch/rustup.py \
    python3 -I /context/fetch/rustup.py 1.29.0 4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10 1.94.1 e7601339027f33d2dce8ec558d05b46a46e8e2c0c72dd30b247f64afc4f087f4 \
    && chmod --recursive a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

LABEL org.opencontainers.image.title="FrontierCS2 verifier egraph_invariant_restoration" \
      dev.frontier.problem-id="egraph_invariant_restoration"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to (with its vendored Cargo dependencies); the generated files; and each
# mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge with this port's evaluator (`eval/evaluator.py`), which checks after every timed
# invocation that it left nothing behind and refuses the brace form of the `std::fs` and
# `std::process` imports.
FROM common AS judge
COPY judge /judge
COPY eval/evaluator.py /judge/evaluator.py
RUN echo "761a895ef41fc19775df5bc190673c7210ccf12a5f584f037a2857efeaebf712  /judge/evaluator.py" | sha256sum --check --strict -

FROM common
COPY --chmod=a+rX,a-w --from=judge /judge /opt/frontier/task
USER 65532:65532
