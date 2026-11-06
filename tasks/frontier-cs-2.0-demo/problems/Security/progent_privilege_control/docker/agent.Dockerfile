# Agent image of progent_privilege_control: the shared Agent base plus this task's Python
# environment (AgentDojo, Progent and the OpenAI client from the pinned lock, the same closure
# as the Verifier), statement, starter repo (with the episode runner and its public harness in
# tools/) and generated submission policy. Composed with the Candidate overlay at Run time, so
# it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libseccomp2 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY repo/tools/requirements-lock.txt /opt/progent/requirements-lock.txt
RUN python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --no-deps -r /opt/progent/requirements-lock.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain"

LABEL org.opencontainers.image.title="FrontierCS2 task progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control"

# The solver-visible tree: the split, the reference measurements, the full upstream tree and
# the attacker-identity generator are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0555 /frontier/task/repo/tools/run_episodes.py \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/progent/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /workspace
