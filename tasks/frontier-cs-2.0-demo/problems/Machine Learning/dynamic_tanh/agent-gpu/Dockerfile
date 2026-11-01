# syntax=docker/dockerfile:1.7
# GPU runner image for r_top01_2503-10622-cand_000-preflight-9740a08b6f00. The platform creates one of these on the
# solver's request and connects it through its own SSH gateway; the solver never
# holds a provider credential and never learns this instance's address.
#
# It carries the toolchain and the read-only source tree, and no part of either
# suite: what runs here is the solver's own build and the task's benchmark, and a
# runner that could see the graded workload would be a second grading path.
# The publisher supplies an immutable image built from FrontierCS2's trusted
# `driver/images/gpu-ssh-base/Dockerfile` recipe.
ARG FRONTIER_GPU_SSH_BASE_IMAGE
FROM ${FRONTIER_GPU_SSH_BASE_IMAGE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libnss-wrapper openssh-client openssh-server python3 python3-dev python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir numpy==2.1.3 pillow==11.0.0 tensorboard==2.18.0 timm==1.0.15 torch==2.5.1 torchvision==0.20.1 zstandard


RUN install -d -o 10001 -g 10001 /workspace

# The public flower_photos archive named in the statement, extracted (ImageFolder
# layout) for local runs; the GPU runner has no network access.
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', '/tmp/flower_photos.tgz')" \
    && echo "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c  /tmp/flower_photos.tgz" | sha256sum -c - \
    && install -d /opt/frontier/public \
    && tar -xzf /tmp/flower_photos.tgz -C /opt/frontier/public --no-same-owner \
    && rm /tmp/flower_photos.tgz \
    && chmod -R a+rX,a-w /opt/frontier/public/flower_photos


# Keep the immutable version label from invalidating reusable dependency/model
# layers when a task source fingerprint changes.
LABEL org.opencontainers.image.title="FrontierCS2 GPU runner r_top01_2503-10622-cand_000-preflight-9740a08b6f00" \
      dev.frontier.problem-id="r_top01_2503-10622-cand_000-preflight-9740a08b6f00" \
      dev.frontier.generator="frontiersmith2"

COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime

# Snapshot only public toolchain/model settings, never the build environment's
# credentials. SSH has a deliberately minimal environment.
RUN ln -s "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime

RUN find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} +



# The trusted bootstrap starts as root, writes root-owned sshd material, and
# OpenSSH executes task commands as the fixed `frontier` account.
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
