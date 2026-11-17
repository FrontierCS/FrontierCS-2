# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): CUDA
# 12.4.1 with cuDNN, devel, on Ubuntu 22.04, pinned by digest, with the source's Python 3.10
# packages from its hash-locked list, the l-DeTok checkpoint, and the auxiliary model assets. It
# is built once for the agent's image and the verifier's, which build on it.

# The l-DeTok checkpoint (3.04 GB) at the revision the source's `model-assets.json` pins,
# fetched by its SHA-256 in a stage of its own, and copied below with `COPY --link`, so that its
# layer depends on its bytes alone and an edit to any other step neither fetches nor pushes it
# again. So is the VGG16 backbone of LPIPS (553 MB), which the source fetches by torch.hub with
# the 8 hexadecimal digits of its name as the only check; this is the file's full SHA-256.
FROM scratch AS detok-checkpoint
ADD --link --checksum=sha256:8e0790fd04a2f70db764db0ecea56ad844a2587062eded1ce8c7b34a654638df https://huggingface.co/jjiaweiyang/l-DeTok/resolve/8e6475697ac1505fe39d35221836f55e4fbc6ce5/detok-BB-gamm3.0-m0.7.pth /detok-BB-gamm3.0-m0.7.pth

FROM scratch AS vgg16-weights
ADD --link --checksum=sha256:397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 https://download.pytorch.org/models/vgg16-397923af.pth /vgg16-397923af.pth

FROM nvidia/cuda@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4 AS runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages (Python, pip, git and tar), from one dated Ubuntu archive snapshot
# (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's CUDA repository.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh ca-certificates git python3 python3-dev python3-pip tar

# The source's list of packages, each file pinned by SHA-256
# (`docker/dependency-base/runtime-layer.lock`, which also names the one wheel that is not on an
# index, copied beside it, and takes PyTorch's CUDA 12.4 wheels from PyTorch's index), and
# zstandard, which the shared verifier imports: the source installs it unpinned at 0.25.0
# (`pins/constraints.txt`); `pins/requirements.txt` is `uv pip compile` of it with the hashes. pip
# installs nothing that a hash in these files does not name.
COPY docker/dependency-base/dependencies /opt/frontier/dependencies
COPY docker/dependency-base/runtime-layer.lock docker/dependency-base/SHA256SUMS /opt/frontier/dependencies/
COPY pins/requirements.txt /opt/frontier/dependencies/zstandard.lock
RUN pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --requirement /opt/frontier/dependencies/runtime-layer.lock \
        --requirement /opt/frontier/dependencies/zstandard.lock

# The public model assets, read-only: the source's three small files (the LFS files of
# `docker/dependency-base/assets`: the LPIPS head, ConvNeXt-S and the Inception weights of
# torch-fidelity), the checkpoint and the VGG16 backbone.
COPY --link --chmod=a+rX,a-w docker/dependency-base/assets /opt/frontier/models
COPY --link --chmod=a+rX,a-w --from=detok-checkpoint /detok-BB-gamm3.0-m0.7.pth /opt/frontier/models/detok/detok-BB-gamm3.0-m0.7.pth
COPY --link --chmod=a+rX,a-w --from=vgg16-weights /vgg16-397923af.pth /opt/frontier/models/torch/hub/checkpoints/vgg16-397923af.pth
# The source checks the wheel, the lock and the three small assets against its `SHA256SUMS`;
# the two large files were checked as they were fetched.
RUN cd / && sha256sum --check --strict /opt/frontier/dependencies/SHA256SUMS \
    && chmod --recursive a+rX,a-w /opt/frontier/dependencies \
    && ls /opt/frontier/models/torch/hub/checkpoints | sort | tr '\n' ' ' \
    | grep --line-regexp 'convnext_small-0c510722.pth vgg16-397923af.pth weights-inception-2015-12-05-6726825d.pth '

# The source's environment: offline models, no telemetry. The image's python3 is the
# interpreter of the judge and of the agent's checks.
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models TORCH_HOME=/opt/frontier/models/torch \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PIP_DISABLE_PIP_VERSION_CHECK=1
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import torch, torchvision, torch_fidelity, diffusers, zstandard; assert torch.__version__ == "2.6.0+cu124", torch.__version__'
