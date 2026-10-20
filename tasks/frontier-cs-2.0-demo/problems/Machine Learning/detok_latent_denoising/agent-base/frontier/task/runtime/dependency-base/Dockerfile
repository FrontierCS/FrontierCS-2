# Local build allocated by inbox df7a46033b4e43139400cba9f563f0f5; no publication.
# Context: $FRONTIERSMITH_WORK/runtime-staging/exporter-input
FROM nvidia/cuda@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates git tar openssh-client \
    && rm -rf /var/lib/apt/lists/*
COPY dependencies/ /opt/frontier/dependencies/
COPY runtime-layer.lock /opt/frontier/dependencies/runtime-layer.lock
RUN python3 -m pip install --no-cache-dir --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r /opt/frontier/dependencies/runtime-layer.lock

# Only assets unsupported by the current exporter enter this private base.
# The exporter subsequently stages the pinned HF tokenizer and supported VGG16.
COPY assets/ /opt/frontier/models/
COPY SHA256SUMS /opt/frontier/dependencies/SHA256SUMS
RUN cd / && sha256sum -c /opt/frontier/dependencies/SHA256SUMS \
    && chmod -R a+rX,a-w /opt/frontier/models /opt/frontier/dependencies
ENV TORCH_HOME=/opt/frontier/models/torch \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY record_image_packages.py /opt/frontier/build-evidence/record_image_packages.py
RUN useradd --uid 10001 --no-create-home --home-dir /workspace --shell /usr/sbin/nologin frontier \
    && python3 /opt/frontier/build-evidence/record_image_packages.py \
    && chmod -R a+rX,a-w /opt/frontier/build-evidence
