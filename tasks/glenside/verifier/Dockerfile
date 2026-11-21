ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bash build-essential ca-certificates cargo git patch python3 python3-pip rustc tar util-linux xz-utils \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt
# Rust 1.85.1 as the source installs it, rustup 1.28.1 and the minimal profile under /usr/local, but
# with the rustup-init file and the installed toolchain each pinned by a digest (`fetch/rustup.py`).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUST_VERSION=1.85.1 PATH=/usr/local/cargo/bin:${PATH}
RUN --mount=type=bind,source=fetch/rustup.py,target=/context/fetch/rustup.py \
    python3 -I /context/fetch/rustup.py 1.28.1 a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f 1.85.1 638f4b5eb8d14aee8a3a1a4e3c9410c6a40bfb37452d7a4eec000b8ce12a4f69 \
    && chmod --recursive a+w /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version
# The Cargo dependencies of the starter, offline from here on: `--locked` makes Cargo fetch what
# `Cargo.lock` names, each crates.io package checked against the lock's checksum and each git
# dependency taken at the lock's commit.
COPY repo/code /opt/frontier-cargo-input
RUN CARGO_HOME=/opt/cargo cargo fetch --locked --manifest-path /opt/frontier-cargo-input/Cargo.toml \
    && chmod --recursive a+rX /opt/cargo
ENV CARGO_HOME=/opt/cargo CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
# The dependency build the judge starts from: the starter built offline once, and a dependency build
# seed that every judge build copies into its isolate instance. Cargo fingerprints include the
# dependency source paths, so the seed is built against the CARGO_HOME the instances read in place
# (/opt/cargo); the target dir itself may move. Dependencies only, never a Glenside answer.
COPY repo /opt/frontier-public
RUN python3 /opt/frontier-public/frontier-build.py --output /opt/frontier-prebuild --target-dir /opt/frontier-dependency-target \
    && rm --recursive --force /opt/frontier-dependency-target /opt/frontier-cargo-input /opt/frontier-public
RUN CARGO_TARGET_DIR=/opt/frontier-cargo-seed CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 \
      cargo test --manifest-path /opt/frontier-prebuild/code/Cargo.toml --locked --offline --no-default-features --lib --no-run \
    && CARGO_TARGET_DIR=/opt/frontier-cargo-seed \
      cargo clean --manifest-path /opt/frontier-prebuild/code/Cargo.toml --package glenside \
    && chmod --recursive a+rX,a-w /opt/frontier-cargo-seed /opt/cargo

LABEL org.opencontainers.image.title="FrontierCS2 verifier glenside" \
      dev.frontier.problem-id="glenside"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GLENSIDE_WORK_ROOT=/frontier/glenside-work
WORKDIR /frontier

# The final workloads' seeds are for the final evaluation alone. The train judge's
# `evaluator.py` (`train/`) has none, and is laid over the judge in a stage of its own, from
# which the train judge's image copies the judge, so that the seeds are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN echo "10f97a4d6a45b754d585fe3a82f33cf35a768247f3067df19bfd582a1b5b747f  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge with the final workloads' seeds left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
