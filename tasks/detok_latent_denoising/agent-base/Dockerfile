# The source's Agent GPU runner image: the runtime the judge's
# image shares (`runtime/Dockerfile`: CUDA 12.4.1, Python 3.10, PyTorch 2.6, the task's hash-locked
# packages and the public model assets), with the solver-visible tree. It holds git, python3 and
# tar, which the submission tools need. The agent runs as uid 1000, which owns /workspace.
FROM runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The starter's `frontier-bench.sh` runs `python`, which the image names `python3` only.
RUN install --directory --owner=1000 --group=1000 /workspace \
    && ln --symbolic /usr/bin/python3 /usr/local/bin/python \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu

LABEL org.opencontainers.image.title="FrontierCS2 task detok_latent_denoising" \
      dev.frontier.problem-id="detok_latent_denoising"

# The solver-visible tree, and nothing else: the evaluator, the reference and both suites are in
# the verifier's image only. The submission tools are linked from /frontier/task as in the
# shared agent base.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
