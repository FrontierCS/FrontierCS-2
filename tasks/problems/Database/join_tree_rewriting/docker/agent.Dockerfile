# Agent image of join_tree_rewriting: the shared Agent base plus this task's statement, starter
# repo and generated submission policy (the base provides git and Python 3; pip and the
# `python` command are added). Composed
# with the Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT, CMD or
# named user. Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

# The old image (python:3.11-slim) also had `python` and `pip`; keep both commands.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task join_tree_rewriting" \
      dev.frontier.problem-id="join_tree_rewriting"

# The solver-visible tree, and nothing else: the evaluator, reference, data, executor and
# beat-reference specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
