ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages. The agent runs as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates fplll-tools git libgmp-dev libmpfr-dev \
           pkg-config python3 python3-dev \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task lwe_structured_recovery" \
      dev.frontier.problem-id="lwe_structured_recovery"

# The solver-visible tree, and nothing else: the evaluator and reference are in the verifier's
# image only. The repository's submission tools replace the base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
