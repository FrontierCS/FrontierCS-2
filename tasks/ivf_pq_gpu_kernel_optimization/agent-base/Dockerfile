# The source's Agent GPU runner base, PyTorch 2.6.0 with CUDA 12.4 and Triton, pinned by digest:
# the toolchain that the task's checks run with, in the agent's own box, which has the GPU.
# SIFT-128 of ANN-Benchmarks (`sift-128-euclidean.hdf5`, originally
# https://ann-benchmarks.com/sift-128-euclidean.hdf5), from FrontierCS's copy of it, by that
# repository's commit and the file's SHA-256.
FROM scratch AS sift
ADD --checksum=sha256:dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/ec31a694bcf459ab00dc075674df734d81fab56e/ann-benchmarks/sift-128-euclidean.hdf5 /sift.hdf5

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The submission tools need git and tar: the base holds tar and ca-certificates, so git is the one
# package installed. apt is set to one dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`),
# which replaces the base's archive and NVIDIA's CUDA repository, so that a rebuild installs the
# same version. The agent runs as uid 1000, user `ubuntu` as in the shared agent base, which owns
# /workspace.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

# h5py at the version the source pins, each of its files pinned by SHA-256 (`pins/requirements.in`
# is the pin and `pins/requirements.txt` is `uv pip compile --no-deps --generate-hashes` of it:
# the base holds the numpy it needs). The SIFT-128 data are the stage `sift`'s, checked by SHA-256,
# which the judge's image holds too; the public benchmark is the starter's own `benchmark/`, where the source's runner
# image keeps it.
RUN --mount=type=bind,source=pins/requirements.txt,target=/context/pins/requirements.txt \
    python3 -m pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /context/pins/requirements.txt
COPY --link --chmod=a+rX,a-w --from=sift /sift.hdf5 /data/sift.hdf5
RUN echo "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984  /data/sift.hdf5" | sha256sum --check --strict -
COPY --chmod=a+rX,a-w repo/benchmark /opt/frontier/development

LABEL org.opencontainers.image.title="FrontierCS2 task ivf_pq_gpu_kernel_optimization" \
      dev.frontier.problem-id="ivf_pq_gpu_kernel_optimization"

# The solver-visible tree, and nothing else: the evaluator, the reference and the final
# workloads are in the verifier's image only. The submission tools are linked from
# /frontier/task as in the shared agent base.
COPY --from=task-tools --chmod=0555 /agent/bin/make_submission.sh /agent/bin/prepare_workspace.sh /frontier/task/bin/
COPY --from=task-tools --chmod=0444 /agent/bin/fcs2_submission.py /lib/edit_scope.py /frontier/task/bin/
COPY --chmod=0444 generated/agent/submission-policy.json /frontier/task/bin/
COPY --chmod=a+rX,a-w --from=readme /README.md /frontier/task/README.md
COPY --chmod=a+rX,a-w repo /frontier/task/repo
RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
