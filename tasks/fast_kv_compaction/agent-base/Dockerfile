# The source's Agent GPU runner base, CUDA 12.8.1 devel on Ubuntu 24.04 with the source's Python
# environment and Qwen3-4B, built once for this image and the verifier's (`runtime/Dockerfile`):
# the toolchain that the starter's checks import, and the weights. It holds git,
# python3 and tar, which the submission tools need. The agent runs as uid 1000, the base image's
# `ubuntu`, which owns /workspace.
FROM runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction"

# The solver-visible tree, and nothing else: the evaluator, the reference tree, the frozen subset
# and the reference targets are in the verifier's image only. The submission tools are the
# repository's `shared/` ones, which the build puts in its context, linked from /frontier/task as
# the shared agent base links them.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
