# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000, the base
# image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 tar \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task hypergraph_core_numbers" \
      dev.frontier.problem-id="hypergraph_core_numbers"

# The solver-visible tree, and nothing else: the evaluator, reference and graphs are in the
# verifier's image only. The submission tools are the repository's `shared/` ones, which the
# build puts in its context, and which replace the base's.
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
