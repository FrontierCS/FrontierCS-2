# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them (its list is `requirements.in`, and so is the
# list of the setuptools wheel in `wheels.in`), each file pinned by SHA-256: `requirements.txt` and
# `wheels.txt` are `uv pip compile --generate-hashes` of them, which `pip` installs, or downloads,
# under `--require-hashes`.
COPY agent-base/requirements.txt agent-base/wheels.txt /tmp/
# The task's packages, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000, the base
# image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bash ca-certificates git python3 python3-pip tar \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task residual_state_advancement" \
      dev.frontier.problem-id="residual_state_advancement"

# The solver-visible tree, and nothing else: the evaluator, the reference, the baseline and
# the final data are in the verifier's image only. The submission tools are the repository's
# `shared/` ones, which the build puts in its context, and which replace the base's.
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
    && chmod 0444 /frontier/task/README.md

# An offline setuptools wheel for `pip install -e` of the starter (no network in the sandbox).
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 download --no-cache-dir --require-hashes --only-binary=:all: --no-deps --dest /opt/frontier/wheels --requirement /tmp/wheels.txt \
    && rm /tmp/wheels.txt \
    && chmod --recursive a+rX,a-w /opt/frontier/wheels

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_FIND_LINKS=/opt/frontier/wheels PIP_BREAK_SYSTEM_PACKAGES=1 LANG=C.UTF-8
WORKDIR /workspace
