# The source's Agent GPU runner base, CUDA 12.4.1 with cuDNN on Ubuntu 22.04 with the source's
# Python environment and the offline models, built once for this image and the verifier's
# (`runtime/Dockerfile`): the toolchain that the starter's checks run with, in the agent's own box,
# which has the GPU. It holds git, python3 and tar, which the submission tools need. The agent
# runs as uid 1000, user `ubuntu` as in the shared agent base, which owns /workspace.
FROM runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task representation_frechet_loss" \
      dev.frontier.problem-id="representation_frechet_loss"

# The wrapper that runs the statement's commands in the toolchain's environment, with the
# environment it records now (`gpu/task_runtime.py`).
COPY --chmod=0555 gpu/task_runtime.py /opt/frontier/task-runtime
RUN /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot

# The solver-visible tree, and nothing else: the evaluator, the reference and the final
# workloads are in the verifier's image only. The submission tools are linked from
# /frontier/task as in the shared agent base.
COPY --from=task-tools --chmod=0555 /agent/bin/make_submission.sh /agent/bin/prepare_workspace.sh /frontier/task/bin/
COPY --from=task-tools --chmod=0444 /agent/bin/fcs2_submission.py /lib/edit_scope.py /frontier/task/bin/
COPY --chmod=0444 generated/agent/submission-policy.json /frontier/task/bin/
COPY --chmod=a+rX,a-w --from=readme /README.md /frontier/task/README.md
COPY --chmod=a+rX,a-w repo /frontier/task/repo
RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
