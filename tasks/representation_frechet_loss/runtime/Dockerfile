# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): CUDA
# 12.4.1 with cuDNN on Ubuntu 22.04, pinned by digest, with the source's Python 3.10 packages from
# the hashed lock and the offline model snapshots, built once for the agent's image and the
# verifier's, which build on it.

# Each model file is fetched by its SHA-256 (the digests of `judge/judge_deps/assets.json`, which
# the judge checks again before it reads them) in a stage of its own, and the runtime copies each
# stage's files with `COPY --link`, so that a layer depends on its files' bytes alone and an edit to
# any other step of this recipe neither fetches nor pushes them again.
FROM scratch AS fd-loss
ADD --link --checksum=sha256:aa95e128b3378e25285f9c2470d1e074d73a8d9d7846a7c9ab59e851c2ed64f8 https://huggingface.co/jjiaweiyang/FD-Loss/resolve/88ce2bb4061b0dd58839a1827235ffcffa51eed2/checkpoints/base/pMF-B_256.pth /checkpoints/base/pMF-B_256.pth
ADD --link --checksum=sha256:fe994d2be02d97b7494c393e0e4b283142746e18be48a5a34fc0a8f415748da3 https://huggingface.co/jjiaweiyang/FD-Loss/resolve/88ce2bb4061b0dd58839a1827235ffcffa51eed2/data/fid_stats/paper_ref_stats.pkl /data/fid_stats/paper_ref_stats.pkl

FROM scratch AS convnextv2
ADD --link --checksum=sha256:f56a6b783f771bddfc3c9dee34d211e2ca17a9281887859753c7293f71020d7a https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k/resolve/9e250ed8f88b436472d2b24af26f82a8aa8c719d/config.json /config.json
ADD --link --checksum=sha256:3b04d2fd7f5268b706210542bd6b6833ad75d6a5e4612a1ae34e8414dbd50b81 https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k/resolve/9e250ed8f88b436472d2b24af26f82a8aa8c719d/model.safetensors /model.safetensors

FROM scratch AS clip
ADD --link --checksum=sha256:8a5bb1350acbf8b9157d977ac705f48e4bc9d86d36d0f8fd4e36d11e5ef9cf33 https://huggingface.co/timm/vit_large_patch14_clip_224.openai/resolve/18d0535469bb561bf468d76c1d73aa35156c922b/config.json /config.json
ADD --link --checksum=sha256:eb70c15b117effa76315db43fcbd9f32a655292fc861798343939a02f561c6ae https://huggingface.co/timm/vit_large_patch14_clip_224.openai/resolve/18d0535469bb561bf468d76c1d73aa35156c922b/pytorch_model.bin /pytorch_model.bin

FROM scratch AS dinov2
ADD --link --checksum=sha256:f95b0f718b06aa4c6394afe93840f04e445bf5909f0b021aecbe23f6a55954df https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m/resolve/4741e1cafbf45415e77074bb0cb42dba76c8684a/config.json /config.json
ADD --link --checksum=sha256:0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355 https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m/resolve/4741e1cafbf45415e77074bb0cb42dba76c8684a/model.safetensors /model.safetensors

FROM scratch AS mae
ADD --link --checksum=sha256:568d0f93881f821a5c12faefb7f9d4e4190c47c17680a8aee64ce98b3a18dd3a https://huggingface.co/timm/vit_large_patch16_224.mae/resolve/126727e9b8ccd632d08a0d75fff095017c225e54/config.json /config.json
ADD --link --checksum=sha256:d58a1a547a07c3c41917c23069dcaadab6f469a8cf793a3cba5fe404c344ab68 https://huggingface.co/timm/vit_large_patch16_224.mae/resolve/126727e9b8ccd632d08a0d75fff095017c225e54/model.safetensors /model.safetensors

FROM scratch AS siglip2
ADD --link --checksum=sha256:6253f3cd27dc62e6f551cadabe6de3fb94d9d85d6ccda03b9e8bf1ed6876c29d https://huggingface.co/timm/vit_so400m_patch16_siglip_256.v2_webli/resolve/a8dedb158968170d6fb45a000e0d5755d8021386/config.json /config.json
ADD --link --checksum=sha256:68c976f5691fd6a20187da6b3c6132601dba52242c349acf8534c64b99ab2b96 https://huggingface.co/timm/vit_so400m_patch16_siglip_256.v2_webli/resolve/a8dedb158968170d6fb45a000e0d5755d8021386/model.safetensors /model.safetensors

FROM scratch AS inception
ADD --link --checksum=sha256:6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2 https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth /hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

# The source's Verifier and Agent GPU runner base, by digest.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages (git, tar, Python and pip; the base holds the CA certificates) from one
# dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that replaces the base's archive and
# NVIDIA's CUDA repository, which no snapshot covers; none of it comes from that repository. Then
# the source's pip packages, each file pinned by SHA-256 (`pins/requirements.in` is the source's
# list, `pins/requirements.txt` is `uv pip compile --generate-hashes` of it under the source's
# constraints, which `pip` installs under `--require-hashes`).
COPY pins/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh ca-certificates git python3 python3-pip tar \
    && pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory /opt/frontier \
    && ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python

# The released checkpoint and statistics, the five representation models and the Inception
# weights, read-only and offline from here on, at the revisions the source pins.
COPY --link --chmod=a+rX,a-w --from=fd-loss / /opt/frontier/models/fd-loss/
COPY --link --chmod=a+rX,a-w --from=mae / /opt/frontier/models/mae/
COPY --link --chmod=a+rX,a-w --from=siglip2 / /opt/frontier/models/siglip2/
COPY --link --chmod=a+rX,a-w --from=convnextv2 / /opt/frontier/models/convnextv2/
COPY --link --chmod=a+rX,a-w --from=dinov2 / /opt/frontier/models/dinov2/
COPY --link --chmod=a+rX,a-w --from=clip / /opt/frontier/models/clip/
COPY --link --chmod=a+rX,a-w --from=inception / /opt/frontier/models/torch/
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/opt/frontier/models/torch
