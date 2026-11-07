# Verifier control image of vllm_llm_serving_optimization: the shared Verifier base plus this
# task's standalone Verifier (verifier/verify.py, task.toml [verifier] command) and judge. It
# drives the Driver-managed H100 parents (docker/verifier-gpu.Dockerfile), the scheduler service
# and the SWE-bench test sandboxes; the packages and the clean vLLM checkout are those of the
# previous release.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The task runs its own Verifier at /opt/frontier/verifier/ ([verifier] command); the base's
# shared entry script of that name is not used here and would block the directory.
RUN rm /opt/frontier/verifier \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libnss-wrapper openssh-client python3 python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o 65532 -g 65532 /frontier
COPY verifier/requirements.txt /opt/frontier/requirements.txt
RUN python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/pip install --no-cache-dir -r /opt/frontier/requirements.txt
# The clean checkout the patch is checked against: repo/ with vLLM's own README.md (repo/README.md
# is the task statement), so the tree is exactly upstream's.
COPY repo /opt/frontier/vllm-clean
COPY docker/vllm-README.md /opt/frontier/vllm-clean/README.md
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
COPY --chmod=0444 verifier/runtime-config.json /opt/frontier/runtime-config.json
COPY --chmod=0444 verifier/runtime-plan.json /opt/frontier/runtime-plan.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
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
       /opt/frontier/controller-contract.json /opt/frontier/controller-contract-limits.json \
       /opt/frontier/effective-task.toml

LABEL org.opencontainers.image.title="FrontierCS2 verifier control vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/opt/frontier/judge:/opt/frontier/verifier \
    HOME=/frontier TMPDIR=/frontier
USER 65532:65532
WORKDIR /frontier
ENTRYPOINT []
CMD ["/opt/frontier/python/bin/python", "/opt/frontier/verifier/verify.py"]
