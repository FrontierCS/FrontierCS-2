# The source's Verifier GPU runner base, CUDA 12.8.0 devel on Ubuntu 24.04, pinned by digest,
# where the judge compiles with nvcc and runs the candidate's code under `isolate` on the GPU.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48 AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages and zstandard for the verifier's evidence, at the version the source pins,
# each of its files pinned by SHA-256 (`verifier/requirements.in` is the pin and
# `verifier/requirements.txt` is `uv pip compile --generate-hashes` of it, which `pip` installs
# under `--require-hashes`); and the user 65532 with /frontier its home, as the shared verifier base
# has them. The base holds bash, ca-certificates and tar. The rest is installed from one dated
# Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's
# CUDA repository.
COPY verifier/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier kami_fp64_block_gemm" \
      dev.frontier.problem-id="kami_fp64_block_gemm"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context.
# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launchers are those of the shared verifier base.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     shared/lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The hidden workloads' shapes and seeds are for the final evaluation alone. The train judge's
# `evaluator.py` (`train/`) has none, and is laid over the judge in a stage of its own, from
# which the train judge's image copies the judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/evaluator.py /judge/evaluator.py
RUN echo "107c11860fe24e2262a6b590a30146b9d6eb754f35cd6b73ed406f10e5400fca  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge with the hidden workloads left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
