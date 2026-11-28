ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The source's packages less `openssh-client`, which served its SSH runner, from the same Ubuntu
# 24.04 snapshot. The agent runs as uid 1000, the base image's `ubuntu`, which owns /workspace. Its
# H100 runner is `gpu/Dockerfile`.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bash build-essential ca-certificates curl git python3 python3-pip \
       python3-venv ripgrep tar \
    && install --directory --owner=1000 --group=1000 /workspace

# vLLM 0.11.0, as the full ID of upstream's commit, which names its content.
ARG VLLM_COMMIT=b8b302cde434df8c9289a2b465406b47ebab1c2d

LABEL org.opencontainers.image.title="FrontierCS2 task vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization" \
      dev.frontier.vllm-commit="${VLLM_COMMIT}"

# The solver-visible tree, and nothing else: the statement and the clean vLLM checkout, which is
# upstream's tree at that commit, fetched from GitHub, with the task's `repo/` over it: the
# statement, in place of vLLM's own README.md. The repository's submission tools replace the
# base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
RUN git init --quiet /frontier/task/repo \
    && git -C /frontier/task/repo fetch --quiet --depth 1 https://github.com/vllm-project/vllm "$VLLM_COMMIT" \
    && git -C /frontier/task/repo checkout --quiet "$VLLM_COMMIT" \
    && rm --recursive --force /frontier/task/repo/.git
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100
WORKDIR /workspace
