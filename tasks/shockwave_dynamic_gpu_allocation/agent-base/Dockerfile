ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The simulator's Python packages, as the source pins them, with the packages they need, each
# file pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY agent-base/requirements.txt /tmp/requirements.txt
# The task's packages, from the same Ubuntu 24.04 snapshot, then its Python packages. The agent runs
# as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python-is-python3 python3 python3-pip python3-venv tar \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation"

# The solver-visible tree, and nothing else: the evaluator, traces, reference and beat-reference
# specification are in the verifier's image only. The repository's submission tools replace the
# base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
