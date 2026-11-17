# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): the
# PyTorch 2.4.1 / CUDA 12.4 runtime image, pinned by digest, with the source's Python packages from
# a hashed lock and the offline models, built once for the agent's image and the verifier's, which
# build on it.

# Each model file is fetched by its SHA-256 in a stage of its own and copied below with `COPY
# --link`, so that a layer depends on its files' bytes alone. The SHA-256 of the two large files
# and of the VAE's two weight files are those of the source's `model-assets.json`; the others are
# the SHA-256 of the file's bytes at that revision, whose git blob ID that file gives and
# equals (checked when the lock was written).
FROM scratch AS point-maze
# knightnemo/nanowm-b2-dino-wm-point-maze-30k @ b84303b7151cbd1836841c3a87bf095e1015b150
ADD --link --checksum=sha256:11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361 https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k/resolve/b84303b7151cbd1836841c3a87bf095e1015b150/.gitattributes /.gitattributes
ADD --link --checksum=sha256:f13f836da91323ba6fdc53bac23d60afa427e70a347584338602e0b4f7df115b https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k/resolve/b84303b7151cbd1836841c3a87bf095e1015b150/README.md /README.md
ADD --link --checksum=sha256:e621080ad71fb9c1fe14f0874652c276347f42c19d0a020d0c1732867cf9c1a3 https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k/resolve/b84303b7151cbd1836841c3a87bf095e1015b150/config.yaml /config.yaml
ADD --link --checksum=sha256:86ac17ff150da027618476817ca48708014de6f5d8d09277fe352e2584f54f28 https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k/resolve/b84303b7151cbd1836841c3a87bf095e1015b150/model.safetensors /model.safetensors

FROM scratch AS sd-vae
# stabilityai/sd-vae-ft-mse @ 31f26fdeee1355a5c34592e401dd41e45d25a493
ADD --link --checksum=sha256:15bff3ac97756db161d7bae5a5aa24899c4c01fc02effcc3f0499cb642a3618b https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/31f26fdeee1355a5c34592e401dd41e45d25a493/.gitattributes /.gitattributes
ADD --link --checksum=sha256:8ac69483c18a45d0fee531e08b6ede97ee9b4c79e875659c1b0e6a5903ca3400 https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/31f26fdeee1355a5c34592e401dd41e45d25a493/README.md /README.md
ADD --link --checksum=sha256:92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/31f26fdeee1355a5c34592e401dd41e45d25a493/config.json /config.json
ADD --link --checksum=sha256:1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/31f26fdeee1355a5c34592e401dd41e45d25a493/diffusion_pytorch_model.bin /diffusion_pytorch_model.bin
ADD --link --checksum=sha256:a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815 https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/31f26fdeee1355a5c34592e401dd41e45d25a493/diffusion_pytorch_model.safetensors /diffusion_pytorch_model.safetensors

# The VGG16 weights that LPIPS loads, by the full SHA-256 of the file, where the source checks only
# the eight hex digits of its name (`check_hash=True`).
FROM scratch AS vgg16
ADD --link --checksum=sha256:397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 https://download.pytorch.org/models/vgg16-397923af.pth /hub/checkpoints/vgg16-397923af.pth

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages less the SSH ones and
# `libnss-wrapper`, which serve only its SSH runner, from one dated Ubuntu archive snapshot
# (`fetch/apt_snapshot.sh`) that replaces the base's archive. Then the source's pip packages, each
# file pinned by SHA-256 (`pins/requirements.in` is the source's list; `pins/requirements.txt` is
# what `pip install --dry-run --report` installs for it under the source's constraints on this base,
# so the packages the base holds, `torch`, `torchvision` and `setuptools` among them, stay as they
# are, as under the source's `pip3 install`), which `pip` installs under `--require-hashes`; two of
# its packages (`antlr4-python3-runtime`, `moviepy`) have an sdist alone, which is built with the
# base's own setuptools and wheel and no build isolation. The conda Python of the base is the one on
# the path, as for the source's `pip3`.
COPY pins/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh bash ca-certificates git libseccomp2 python3 python3-pip tar \
    && pip3 install --no-cache-dir --require-hashes --no-build-isolation --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python3 -c 'import torch; assert torch.__version__.startswith("2.4.1"), torch.__version__' \
    && install --directory /opt/frontier \
    && printf '#!/bin/sh\nexec %s "$@"\n' "$(python3 -c 'import sys; print(sys.executable)')" > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python

# The pinned checkpoint and the VAE, and the VGG16 weights, read-only and offline from here on.
COPY --link --chmod=a+rX,a-w --from=point-maze / /opt/frontier/models/point-maze/
COPY --link --chmod=a+rX,a-w --from=sd-vae / /opt/frontier/models/sd-vae/
COPY --link --chmod=a+rX,a-w --from=vgg16 / /opt/frontier/models/torch/
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/opt/frontier/models/torch
