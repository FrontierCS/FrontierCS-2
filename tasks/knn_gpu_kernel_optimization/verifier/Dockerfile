# The source's Verifier GPU runner base, PyTorch 2.6.0 with CUDA 12.4 and Triton, pinned by
# digest, where the judge runs every implementation under `isolate` on the GPU.
# SIFT-128 of ANN-Benchmarks (`sift-128-euclidean.hdf5`, originally
# https://ann-benchmarks.com/sift-128-euclidean.hdf5), from FrontierCS's copy of it, by that
# repository's commit and the file's SHA-256.
FROM scratch AS sift
ADD --checksum=sha256:dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/ec31a694bcf459ab00dc075674df734d81fab56e/ann-benchmarks/sift-128-euclidean.hdf5 /sift.hdf5

# GIST-960 of ANN-Benchmarks (`gist-960-euclidean.hdf5`, originally
# https://ann-benchmarks.com/gist-960-euclidean.hdf5), from the same copy.
FROM scratch AS gist
ADD --checksum=sha256:8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/ec31a694bcf459ab00dc075674df734d81fab56e/ann-benchmarks/gist-960-euclidean.hdf5 /gist.hdf5

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

# The datasets are the stages `sift`'s and `gist`'s, checked against `judge/data_manifest.json`
# below; h5py reads them, at the version
# the source pins, each of its files pinned by SHA-256 (`verifier/requirements.in` is the pin and
# `verifier/requirements.txt` is `uv pip compile --no-deps --generate-hashes` of it: the base
# holds the numpy it needs).
COPY verifier/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt
COPY --link --chmod=a+rX,a-w --from=sift /sift.hdf5 /data/sift.hdf5
COPY --link --chmod=a+rX,a-w --from=gist /gist.hdf5 /data/gist.hdf5

LABEL org.opencontainers.image.title="FrontierCS2 verifier knn_gpu_kernel_optimization" \
      dev.frontier.problem-id="knn_gpu_kernel_optimization"

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
    && /opt/frontier/runtime-python -c 'import torch, zstandard, h5py' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /data /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge with this port's evaluator and worker (`eval/`), which compute the quality metrics
# themselves from the outputs that the worker returns, where the source's worker computes them.
# It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py eval/knn_worker.py /judge/
RUN echo "07d1cfcbad937526986eb14a56c3e4a269ce44aee75ad5921ad513ba3a026764  /judge/evaluator.py" | sha256sum --check --strict - \
    && echo "ef89eef08571a95959a5cf64c622feab33d2d3ec4da2a894e31682f7ce3405d3  /judge/knn_worker.py" | sha256sum --check --strict -

# The hidden workloads' datasets, sizes and seed are for the final evaluation alone. The train judge's
# `scoring_config.json` (`train/`) has placeholders for them, and is laid over the judge in a
# stage of its own, from which the train judge's image copies the judge, so that they are in none
# of its layers.
FROM judge-eval AS judge-train
COPY train/scoring_config.json /judge/scoring_config.json
RUN echo "f43382f27b63c5d7e472b8905718d06a1b2acaf6a46feaceab473d37db7783eb  /judge/scoring_config.json" | sha256sum --check --strict -

# The train judge's image: the judge with the hidden workloads left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
RUN python3 -B -c "import sys; sys.path.insert(0, '/opt/frontier/task'); from evaluator import validate_data_manifest; validate_data_manifest('/opt/frontier/task/data_manifest.json', '/data', ['sift', 'gist'])"
USER 65532:65532

# The final judge's image: the judge with this port's evaluator and worker.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
RUN python3 -B -c "import sys; sys.path.insert(0, '/opt/frontier/task'); from evaluator import validate_data_manifest; validate_data_manifest('/opt/frontier/task/data_manifest.json', '/data', ['sift', 'gist'])"
USER 65532:65532
