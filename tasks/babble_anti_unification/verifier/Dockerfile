# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 zlib1g
# Rust 1.64.0 as the source installs it, rustup 1.25.1 and the minimal profile under /usr/local,
# but with
# the rustup-init file and the installed toolchain each pinned by a digest (`fetch/rustup.py`).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUSTUP_TOOLCHAIN=1.64.0 PATH=/usr/local/cargo/bin:${PATH}
RUN --mount=type=bind,source=fetch/rustup.py,target=/context/fetch/rustup.py \
    python3 -I /context/fetch/rustup.py 1.25.1 5cc9ffd1026e82e7fb2eec2121ad71f4b0f044e88bca39207b3f6b769aaa799c 1.64.0 2762856211070ec3850823e439a5df72ab1ade6fcb0bd42975bb4046ba4d0ac7 \
    && chmod --recursive a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

LABEL org.opencontainers.image.title="FrontierCS2 verifier babble_anti_unification" \
      dev.frontier.problem-id="babble_anti_unification"

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

# The final corpora are for the final evaluation alone. The train judge's copy of the judge is
# made without them, in a stage of its own, from which the train judge's image copies it, so
# that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
RUN rm --recursive --force /judge/workloads/final
RUN test ! -e /judge/workloads/final

# The train judge's image: the judge without the final corpora.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
