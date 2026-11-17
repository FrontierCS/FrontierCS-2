# The answering model, Qwen3-4B at the revision the source pins, each file fetched by its SHA-256
# (the source's `model-assets.json` gives it for the four large files, and for the others, LICENSE
# too, it is the digest of the file at that revision, whose git blob ID `model-assets.json` gives
# and equals). Its own stage, copied below as a linked layer, so that the layer holds the files
# alone and keeps its digest when an earlier step of the runtime changes.
FROM scratch AS qwen3-4b
ADD --checksum=sha256:8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/config.json /Qwen3-4B/config.json
ADD --checksum=sha256:2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/generation_config.json /Qwen3-4B/generation_config.json
ADD --checksum=sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/LICENSE /Qwen3-4B/LICENSE
ADD --checksum=sha256:8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/merges.txt /Qwen3-4B/merges.txt
ADD --checksum=sha256:328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00001-of-00003.safetensors /Qwen3-4B/model-00001-of-00003.safetensors
ADD --checksum=sha256:6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00002-of-00003.safetensors /Qwen3-4B/model-00002-of-00003.safetensors
ADD --checksum=sha256:e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00003-of-00003.safetensors /Qwen3-4B/model-00003-of-00003.safetensors
ADD --checksum=sha256:6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model.safetensors.index.json /Qwen3-4B/model.safetensors.index.json
ADD --checksum=sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json /Qwen3-4B/tokenizer.json
ADD --checksum=sha256:d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer_config.json /Qwen3-4B/tokenizer_config.json
ADD --checksum=sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 --chmod=0444 \
    https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/vocab.json /Qwen3-4B/vocab.json

# The runtime that FrontierCS's GPU runners of this task share (the judge's and the agent's): CUDA
# 12.8.1 devel on Ubuntu 24.04, pinned by digest, with the source's Python 3.12 venv from the hashed
# wheel lock, installed with no dependency resolution and no unhashed file, and Qwen3-4B at the
# revision the source pins. It is built once for the agent's image and the verifier's, which
# build on it.
FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43

USER root
ARG DEBIAN_FRONTEND=noninteractive
COPY docker/gpu-requirements.lock /opt/frontier/python-lock/requirements.lock
# The base holds bash, ca-certificates, gcc, g++ and tar. The rest is installed from one dated
# Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's
# CUDA repository, which no snapshot covers; none of it comes from that repository. The wheels come
# from PyPI, each checked against the lock.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git iproute2 python3 python3-dev python3-pip python3-venv \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --only-binary=:all: --require-hashes --ignore-installed \
       --index-url https://pypi.org/simple -r /opt/frontier/python-lock/requirements.lock \
    && /opt/frontier/python/bin/python -m pip check
ENV PATH="/opt/frontier/python/bin:${PATH}"
RUN printf '%s\n' '#!/bin/sh' 'export PATH="/opt/frontier/python/bin:$PATH"' \
      'exec /opt/frontier/python/bin/python3 "$@"' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python

# The model, read-only and offline from here on: the files above, their directories read-only too.
COPY --link --from=qwen3-4b /Qwen3-4B /opt/frontier/models/Qwen3-4B
RUN chmod 0555 /opt/frontier/models /opt/frontier/models/Qwen3-4B
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
