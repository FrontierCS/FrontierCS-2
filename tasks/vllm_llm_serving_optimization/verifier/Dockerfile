# The control image of both judges: the shared verifier base with this task's standalone verifier
# (`verifier/verify.py`) and its judge under /opt/frontier/task. It holds no GPU: it leases the H100
# parents of `parent/Dockerfile` through the judge's sessions, and drives the scheduler and the SWE
# containers of each parent's pod.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The verifier's packages, each file pinned by SHA-256 (`requirements.txt` is `uv pip compile` of
# `requirements.in`, the source's list), in a venv of their own; and the source's apt packages
# less those of its SSH client.
COPY verifier/requirements.txt /opt/frontier/requirements.txt
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git python3 python3-venv tar \
    && python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /opt/frontier/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier control vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The clean checkout the patch is checked against: vLLM 0.11.0 at upstream's commit, fetched from
# GitHub by the commit's full ID, which names its content.
ARG VLLM_COMMIT=b8b302cde434df8c9289a2b465406b47ebab1c2d
RUN git init --quiet /opt/frontier/task/vllm-clean \
    && git -C /opt/frontier/task/vllm-clean fetch --quiet --depth 1 https://github.com/vllm-project/vllm "$VLLM_COMMIT" \
    && git -C /opt/frontier/task/vllm-clean checkout --quiet "$VLLM_COMMIT" \
    && git config --system --add safe.directory /opt/frontier/task/vllm-clean \
    && chmod -R a-w /opt/frontier/task/vllm-clean /opt/frontier/python \
    && chmod a-w /opt/frontier/requirements.txt

# The judge and the verifier, each mode's selector, and the reference patch where the reference
# agent reads it.
COPY judge /opt/frontier/task/judge
COPY verifier/*.py verifier/*.json /opt/frontier/task/verifier/
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
COPY judge/reference.patch /opt/frontier/task/reference.patch
# The fixed targets load, the content contract that the CPU controllers bind their records to is
# made, and the launcher that the judge's call runs replaces the base's.
RUN PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/frontier/task/judge:/opt/frontier/task/verifier \
    /opt/frontier/python/bin/python -c \
    "import json; from fixed_target_release import load_release; load_release('/opt/frontier/task/verifier', json.load(open('/opt/frontier/task/verifier/runtime-config.json')))" \
    && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/frontier/task/judge /opt/frontier/python/bin/python \
    /opt/frontier/task/judge/controller_contract_build.py \
    --root /opt/frontier/task \
    --source-tree judge --source-tree verifier \
    --source judge/reference.patch \
    --contract /opt/frontier/task/controller-contract.json \
    --limits /opt/frontier/task/controller-contract-limits.json \
    && printf '#!/bin/sh\nexec /opt/frontier/python/bin/python /opt/frontier/task/verifier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod -R a+rX,a-w /opt/frontier/task /opt/frontier/datasets

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/opt/frontier/task/judge:/opt/frontier/task/verifier \
    HOME=/frontier TMPDIR=/frontier
USER 65532:65532
WORKDIR /frontier
