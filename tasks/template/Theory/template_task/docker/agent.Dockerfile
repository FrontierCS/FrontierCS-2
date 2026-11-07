# Agent image of template_task: the shared Agent base plus this task's packages, statement,
# starter repo and generated submission policy.
#
# Rules (scripts/validate_task_layout.py::check_native_dockerfile):
#   - the final stage is FROM ${AGENT_BASE} (ARG AGENT_BASE, no default): the build injects
#     tasks/shared/bases.json["agent"];
#   - COPY only README.md, repo/ and .frontier-generated/agent/ (nothing from judge/: the
#     evaluator, reference and suites are private to the Verifier image);
#   - no ENTRYPOINT, CMD or named USER: the image is composed with the Candidate overlay;
#   - pin what you install when its version can change the answer's behaviour.
# The base already holds /frontier/task/bin (prepare_workspace.sh, make_submission.sh, the
# shared submission tool), the frontier user (10001) and /workspace.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git python3 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task template_task" \
      dev.frontier.problem-id="template_task"

COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

# The task tree is read-only to the solver; scripts stay executable.
RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
