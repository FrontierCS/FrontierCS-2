# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY agent-base/requirements.txt /tmp/requirements.txt
# The task's packages. The retained parser imports typing_extensions, which Python 3.12 no longer
# pulls in, so `requirements.in` pins it as the old python:3.11 image resolved it. There is no apt
# z3: Ubuntu 24.04 ships Z3 4.8.12, and frontier-build.sh requires 4.13 first on PATH (/usr/bin
# before /usr/local/bin); the z3-solver==4.13.3.0 wheel provides /usr/local/bin/z3. The agent runs
# as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh autoconf automake binutils build-essential ca-certificates curl file git libtool \
           m4 make mariadb-server openssh-client patch pkg-config python-is-python3 python3 \
           python3-dev python3-pip unzip wget \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task polygon" \
      dev.frontier.problem-id="polygon"

# The solver-visible tree, and nothing else: the evaluator, the graded workloads, the reference
# and the beat-reference specification are in the verifier's image only. The submission tools
# are the repository's `shared/` ones, which the build puts in its context, and which replace
# the base's.
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

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
