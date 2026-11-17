# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's Python environment, AgentDojo, Progent and the OpenAI client, in a venv of its own as
# the source makes it, from the source's lock, each file pinned
# by SHA-256 (`requirements.in` is the source's `repo/tools/requirements-lock.txt`;
# `requirements.txt` is `uv pip compile --generate-hashes --no-deps` of it, which `pip` installs
# under `--require-hashes`). The agent runs as uid 1000, the base image's `ubuntu`, which owns
# /workspace. `/opt/progent/no-key` stands in for the key file the episode runner reads
# (`repo/tools/harness/llm_gateway.py:48-55`): the model endpoint holds the key, and drops the
# client's.
COPY agent-base/requirements.txt /opt/progent/requirements.txt
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git libseccomp2 python3 python3-venv \
    && python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /opt/progent/requirements.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain" \
    && printf 'unused\n' > /opt/progent/no-key \
    && chmod 0444 /opt/progent/no-key \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control"

# The solver-visible tree: the split, the reference measurements, the full upstream tree and the
# attacker-identity generator are in the verifier's image only. The submission tools are the
# repository's `shared/` ones, which the build puts in its context.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0555 /frontier/task/repo/tools/run_episodes.py \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/progent/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /workspace
