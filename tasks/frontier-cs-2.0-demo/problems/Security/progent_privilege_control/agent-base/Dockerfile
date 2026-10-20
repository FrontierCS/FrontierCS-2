# syntax=docker/dockerfile:1.7
# Task CPU base for progent_privilege_control. Composed with the Candidate overlay
# at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/):
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control" \
      dev.frontier.generator="frontiersmith2"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libseccomp2 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace

# Same pinned dependency closure as the verifier (AgentDojo, Progent, OpenAI client).
COPY agent-base/frontier/task/requirements-lock.txt /opt/progent/requirements-lock.txt
RUN python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --no-deps -r /opt/progent/requirements-lock.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain"

# The solver-visible tree: the pristine upstream repository, the statement, the
# packaging tools and the public half of the episode harness. The split, the
# reference measurements and the attacker-identity generator are verifier-only.
COPY agent-base/frontier/task /frontier/task

RUN ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin /frontier/task/harness -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh /frontier/task/bin/run_episodes.py \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/progent/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /workspace
