# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): the
# veRL / vLLM image of the source, pinned by digest, with the source's packages and the offline
# Qwen2.5-0.5B-Instruct snapshot, built once for the agent's image and the verifier's, which build
# on it.

# The model, each file fetched by its SHA-256 (the source's `model-assets.json` gives the large
# file's; the others are the SHA-256 of the file's bytes at that revision, whose git blob ID that
# file gives and equals, checked when this recipe was written) in a stage of its own, which the
# runtime copies with `COPY --link`, so that the layer depends on the files' bytes alone.
FROM scratch AS qwen2.5-0.5b-instruct
ADD --link --checksum=sha256:11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/.gitattributes /.gitattributes
ADD --link --checksum=sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE /LICENSE
ADD --link --checksum=sha256:b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/README.md /README.md
ADD --link --checksum=sha256:18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/config.json /config.json
ADD --link --checksum=sha256:e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/generation_config.json /generation_config.json
ADD --link --checksum=sha256:599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/merges.txt /merges.txt
ADD --link --checksum=sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors /model.safetensors
ADD --link --checksum=sha256:c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json /tokenizer.json
ADD --link --checksum=sha256:5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer_config.json /tokenizer_config.json
ADD --link --checksum=sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/vocab.json /vocab.json

# The source's Verifier and Agent GPU runner base, by digest.
FROM hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2@sha256:41dc66525bd551be71cb8abe9c1dd8e93c6bc1620c335038bc256ac4cbc09beb

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages less the SSH ones and `libnss-wrapper`, which serve only its SSH
# runner, from one dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that replaces the base's
# archive. Then the source's pip packages, each file pinned by SHA-256 (`pins/requirements.txt`,
# which holds what pip installs for the source's list under its constraints on this base:
# `pins/requirements.in` is the list), which `pip` installs
# under `--require-hashes`; `gym` has an sdist alone, so the build tools
# (`pins/build-requirements.txt`) install first and the sdist is built with them and no build
# isolation.
COPY pins/build-requirements.txt pins/requirements.txt /tmp/
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh bash ca-certificates git python3 python3-pip tar \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --disable-pip-version-check --require-hashes \
       --only-binary=:all: --no-deps --requirement /tmp/build-requirements.txt \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --disable-pip-version-check --require-hashes \
       --no-build-isolation --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/build-requirements.txt /tmp/requirements.txt \
    && install --directory /opt/frontier \
    && ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python

# The model, read-only and offline from here on.
COPY --link --chmod=a+rX,a-w --from=qwen2.5-0.5b-instruct / /opt/frontier/models/qwen2.5-0.5b-instruct/
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
