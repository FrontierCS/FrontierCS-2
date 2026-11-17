# The source's Verifier GPU runner base, PyTorch 2.6.0 with CUDA 12.4 and Triton, pinned by
# digest, where the judge runs every implementation under `isolate` on the GPU.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge needs git (it applies the patches) and tar: the base holds tar and ca-certificates,
# so git is the one package installed, from one dated Ubuntu archive snapshot
# (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's CUDA repository. Then
# the user 65532 with /frontier its home, as the shared verifier base has them.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git \
    && install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

# h5py reads the data, at the version the source pins, each of its files pinned by SHA-256
# (`pins/requirements.in` is the pin and `pins/requirements.txt` is `uv pip compile --no-deps
# --generate-hashes` of it: the base holds the numpy it needs). The SIFT-128 data are the frozen
# bytes of the source's published verifier image, pinned by digest, and checked by SHA-256, which
# `scoring_config.json` names too.
RUN --mount=type=bind,source=pins/requirements.txt,target=/context/pins/requirements.txt \
    python3 -m pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /context/pins/requirements.txt
COPY --link --chmod=a+rX,a-w --from=docker.io/joyemang33/fcs2-task-verifier-ivf_pq_gpu_kernel_optimization-gpu@sha256:879561b3b3a63030b51df9189a2513e90ab3e2f969d24e97403a9b71986eb9e7 /data/sift.hdf5 /data/sift.hdf5
RUN echo "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984  /data/sift.hdf5" | sha256sum --check --strict -

LABEL org.opencontainers.image.title="FrontierCS2 verifier ivf_pq_gpu_kernel_optimization" \
      dev.frontier.problem-id="ivf_pq_gpu_kernel_optimization"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context.
# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launchers are those of the shared verifier base.
COPY --chmod=0444 shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     shared/lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --chmod=0444 generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import torch, zstandard, h5py' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The hidden workloads' shapes and seed are for the final evaluation alone. The train judge's
# `scoring_config.json` (`train/`) has placeholders for them, and is laid over the judge in a
# stage of its own, from which the train judge's image copies the judge, so that they are in none
# of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/scoring_config.json /judge/scoring_config.json
RUN echo "e4d5daf7c52da3f6d90ec9dfc5c7126d948c797d1de391a964d04907d51d4c23  /judge/scoring_config.json" | sha256sum --check --strict - \
    && ! grep --recursive --quiet --extended-regexp '2026091302|sift_q1k_np|sift_q10k|k1024' /judge

# The train judge's image: the judge with the hidden workloads left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
