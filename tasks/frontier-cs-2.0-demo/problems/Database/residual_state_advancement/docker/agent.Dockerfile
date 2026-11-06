# Agent image of residual_state_advancement: the shared Agent base plus this task's packages,
# statement, starter repo and generated submission policy. Composed with the Candidate overlay
# at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

# The packages of the task's previous image (python:3.12-slim-bookworm): the same
# Python packages, pinned to the versions it had, on the shared base's Python 3.12.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git python3 python3-pip tar \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --no-deps cloudpickle==3.1.2 duckdb==1.5.1 joblib==1.6.0 numpy==2.5.4 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2026.5 PyYAML==6.0.2 scikit-learn==1.7.2 scipy==1.18.1 six==1.17.0 threadpoolctl==3.7.0 tzdata==2026.5 \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit

LABEL org.opencontainers.image.title="FrontierCS2 task residual_state_advancement" \
      dev.frontier.problem-id="residual_state_advancement"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

# An offline setuptools wheel for `pip install -e` of the starter (no network in the sandbox).
RUN pip3 download --no-cache-dir --no-deps --only-binary=:all: --dest /opt/frontier/wheels setuptools==80.9.0 \
    && chmod -R a+rX,a-w /opt/frontier/wheels

USER 10001:10001
# pip installs into the user site as it did in the previous python:3.12 image (no PEP 668 lock).
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_FIND_LINKS=/opt/frontier/wheels \
    PIP_BREAK_SYSTEM_PACKAGES=1 LANG=C.UTF-8
WORKDIR /workspace
