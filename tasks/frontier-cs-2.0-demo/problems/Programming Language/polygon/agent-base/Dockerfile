# syntax=docker/dockerfile:1.7
# Task CPU base for polygon. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM python:3.11@sha256:70e8e937c1da72df1688e883a2108b0feff03381187fa78eca6d9b10df411320

LABEL org.opencontainers.image.title="FrontierCS2 task polygon" \
      dev.frontier.problem-id="polygon" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="unknown"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git mariadb-server python3 z3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 bidict==0.24.1 lark==1.3.1 mo-sql-parsing==11.699.26234 multiprocess==0.70.19 mysql-connector-python==26.7.0 pandas==3.0.6 prettytable==3.18.0 tqdm==4.70.1 ujson==6.0.0 z3-solver==4.13.3.0


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

RUN ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /workspace

# The rest of the repository's requirements.txt, so `pip install -r requirements.txt`
# is satisfied locally; the Agent sandbox has no network access.
USER 0:0
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PySMT==0.9.6 iteration-utilities==0.13.0 matplotlib==3.11.2 psycopg2-binary==2.9.13
USER 10001:10001
