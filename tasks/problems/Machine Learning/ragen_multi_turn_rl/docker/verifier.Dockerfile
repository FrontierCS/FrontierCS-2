# syntax=docker/dockerfile:1.7
# Verifier control image of ragen_multi_turn_rl: the shared Verifier base, the shared GPU adapter
# (gpu-tools), this task's judge and starter, and the starter's Sokoban dependencies. The judge
# runs here and replays the returned episodes (judge/replay.py); the PPO runs of candidate and
# reference go to [test_sandboxes.candidate] runners (task.toml) through run_sandboxes().
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnss-wrapper openssh-client \
    && rm -rf /var/lib/apt/lists/*
# judge/replay.py steps the starter's SimpleSokoban here (the GPU images' versions).
RUN --mount=type=bind,source=docker/verifier.constraints.txt,target=/tmp/frontier-pip-constraints.txt \
    PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --constraint /tmp/frontier-pip-constraints.txt \
    gym gym-notices gym-sokoban imageio matplotlib numpy omegaconf pillow pygame setuptools tqdm

LABEL org.opencontainers.image.title="FrontierCS2 verifier control ragen_multi_turn_rl" \
      dev.frontier.problem-id="ragen_multi_turn_rl"

# The judge and the starter (the shared verifier overlays the submission's allowed files).
COPY judge /opt/frontier/task
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY repo /opt/frontier/task/pristine
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
