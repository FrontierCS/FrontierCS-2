# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libnss-wrapper openssh-client python3 python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 65532 frontier && useradd --uid 65532 --gid 65532 --no-create-home frontier \
    && install -d -o 65532 -g 65532 /frontier
COPY verifier/requirements.txt /opt/frontier/requirements.txt
RUN python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/pip install --no-cache-dir -r /opt/frontier/requirements.txt
COPY agent-base/frontier/task/repo/vllm /opt/frontier/vllm-clean
RUN git -C /opt/frontier/vllm-clean init -q \
    && git -C /opt/frontier/vllm-clean -c core.autocrlf=false add -A -f \
    && GIT_AUTHOR_DATE=2026-10-12T00:00:00Z GIT_COMMITTER_DATE=2026-10-12T00:00:00Z \
       git -C /opt/frontier/vllm-clean -c user.name=frontier -c user.email=frontier@example.invalid -c commit.gpgsign=false \
       commit -q -m 'vLLM upstream b8b302cde434df8c9289a2b465406b47ebab1c2d (tree 43680a9894d43bda79748a713d66b2fca1ff5250)' \
    && test "$(git -C /opt/frontier/vllm-clean rev-parse 'HEAD^{tree}')" = 43680a9894d43bda79748a713d66b2fca1ff5250 \
    && test -z "$(git -C /opt/frontier/vllm-clean status --porcelain --untracked-files=all)" \
    && git config --system --add safe.directory /opt/frontier/vllm-clean \
    && chmod -R a-w /opt/frontier/vllm-clean /opt/frontier/python \
    && chmod a-w /opt/frontier/requirements.txt
COPY judge/ /opt/frontier/judge/
COPY verifier/ /opt/frontier/verifier/
# The runtime projection and plan are rendered from policy.json by
# docs/transfers/vllm_llm_serving_optimization/tools/render_release.py and
# checked against it; they contain no image digest and no credential.
COPY --chmod=0444 verifier/runtime-config.json /opt/frontier/runtime-config.json
COPY --chmod=0444 verifier/runtime-plan.json /opt/frontier/runtime-plan.json
RUN PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/frontier/judge:/opt/frontier/verifier \
    /opt/frontier/python/bin/python -c \
    "import json; from fixed_target_release import load_release; load_release('/opt/frontier/verifier', json.load(open('/opt/frontier/runtime-config.json')))"
RUN PYTHONPATH=/opt/frontier/judge /opt/frontier/python/bin/python \
    /opt/frontier/judge/controller_contract_build.py \
    --root /opt/frontier \
    --source-tree judge --source-tree verifier \
    --source runtime-config.json --source runtime-plan.json \
    --source judge/reference.patch \
    --contract /opt/frontier/controller-contract.json \
    --limits /opt/frontier/controller-contract-limits.json \
    && chmod -R a-w /opt/frontier/judge /opt/frontier/verifier \
    && chmod a-w /opt/frontier/runtime-config.json /opt/frontier/runtime-plan.json \
       /opt/frontier/controller-contract.json /opt/frontier/controller-contract-limits.json
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/opt/frontier/judge:/opt/frontier/verifier \
    HOME=/frontier TMPDIR=/frontier
USER 65532:65532
WORKDIR /frontier
ENTRYPOINT []
CMD ["/opt/frontier/python/bin/python", "/opt/frontier/verifier/verify.py"]
