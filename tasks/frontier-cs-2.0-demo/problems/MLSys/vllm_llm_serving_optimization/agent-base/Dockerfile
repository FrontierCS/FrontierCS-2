# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
LABEL org.opencontainers.image.title="FrontierCS2 vLLM serving agent base" \
      dev.frontier.problem-id="vllm_llm_serving_optimization" \
      dev.frontier.upstream-commit="1bccbadc6bead714d25c546cf954a37bb0791fd6" \
      dev.frontier.vllm-commit="b8b302cde434df8c9289a2b465406b47ebab1c2d"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates \
       curl git openssh-client python3 python3-pip python3-venv ripgrep tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace
COPY agent-base/frontier/task /frontier/task
RUN git -C /frontier/task/repo/vllm init -q \
    && git -C /frontier/task/repo/vllm -c core.autocrlf=false add -A -f \
    && GIT_AUTHOR_DATE=2026-10-12T00:00:00Z GIT_COMMITTER_DATE=2026-10-12T00:00:00Z \
       git -C /frontier/task/repo/vllm -c user.name=frontier -c user.email=frontier@example.invalid -c commit.gpgsign=false \
       commit -q -m 'vLLM upstream b8b302cde434df8c9289a2b465406b47ebab1c2d (tree 43680a9894d43bda79748a713d66b2fca1ff5250)' \
    && test "$(git -C /frontier/task/repo/vllm rev-parse 'HEAD^{tree}')" = 43680a9894d43bda79748a713d66b2fca1ff5250 \
    && test -z "$(git -C /frontier/task/repo/vllm status --porcelain --untracked-files=all)" \
    && chmod -R a-w /frontier/task \
    && chmod 0555 /frontier/task/bin/*.sh \
    && git config --system --add safe.directory /frontier/task/repo/vllm
USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100
WORKDIR /workspace
