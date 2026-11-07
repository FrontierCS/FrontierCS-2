# Agent image of shockwave_dynamic_gpu_allocation: the shared Agent base plus this task's
# packages (the simulator's Python stack, pinned as before), statement, starter repo and
# generated submission policy (python-is-python3 added: the repo scripts call `python`). Composed with the Candidate overlay at Run time, so it must not
# depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python-is-python3 python3 python3-pip python3-venv tar \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit \
    && pip3 install --break-system-packages --no-cache-dir cvxpy==1.9.2 grpcio-tools==1.83.1 grpcio==1.83.1 gurobipy==13.0.3 matrix-completion==0.0.2 mosek==11.2.3 numpy==2.5.2 psutil==7.2.2 scikit-learn==1.9.0 scipy==1.18.1

LABEL org.opencontainers.image.title="FrontierCS2 task shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation"

# The solver-visible tree, and nothing else: the evaluator, traces, reference and
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
