ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The packages that the pinned GCC 7.5 toolchain (`repo/frontier-cxx` unpacks it) needs besides its
# own archive: the system's binutils and C library headers (build-essential) and the libraries of
# its compiler, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000, the base image's
# `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential git libgmp10 libmpc3 libmpfr6 python3 tar util-linux zlib1g \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task parallel_radix_sort" \
      dev.frontier.problem-id="parallel_radix_sort"

# The solver-visible tree, and nothing else: the evaluator, reference patch and final seeds are
# in the verifier's image only. The repository's submission tools replace the base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f \( -name '*.sh' -o -name frontier-cxx \) -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
