# syntax=docker/dockerfile:1.7
# Task CPU base for shockwave_dynamic_gpu_allocation. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_ave-nsdi23"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-pip python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir cvxpy==1.9.2 grpcio-tools==1.83.1 grpcio==1.83.1 gurobipy==13.0.3 matrix-completion==0.0.2 mosek==11.2.3 numpy==2.5.2 psutil==7.2.2 scikit-learn==1.9.0 scipy==1.18.1


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && if [ -f /frontier/task/bin/frontier-gpu ]; then \
         chmod 0555 /frontier/task/bin/frontier-gpu; \
         ln -s /frontier/task/bin/frontier-gpu /frontier/task/frontier-gpu; \
       fi \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /workspace
