# The source's Agent GPU runner base, CUDA 12.6.3 on Ubuntu 24.04 with Python 3.12 and the hash-locked
# JAX / Brax / MuJoCo MJX environment, built once for this image and the verifier's
# (`runtime/Dockerfile`): the toolchain that the starter's public check runs with. It holds git,
# python3 and tar, which the submission tools need. The agent runs as uid 1000, the base image's
# `ubuntu`, which owns /workspace.
FROM runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task mujoco_playground_cpu" \
      dev.frontier.problem-id="mujoco_playground_cpu"

# The solver-visible tree, and nothing else: the evaluator, the private workloads and their
# seeds are in the verifier's image only. The submission tools are linked from /frontier/task as
# in the shared agent base. The statement names the hash-locked requirements at
# /frontier/task/runtime and the public check's launcher at /opt/frontier/task-runtime, which
# snapshots the toolchain's environment once.
COPY --from=task-tools --chmod=a+rX,a-w /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY --chmod=a+rX,a-w generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --chmod=a+rX,a-w --from=readme /README.md /frontier/task/README.md
COPY --chmod=a+rX,a-w repo /frontier/task/repo
COPY --chmod=a+rX,a-w docker/requirements-gpu.lock /frontier/task/runtime/requirements-gpu.lock
COPY --chmod=0555 gpu/task_runtime.py /opt/frontier/task-runtime

RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && chmod 0555 /frontier/task/bin/*.sh \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod a-w /frontier/task /frontier/task/bin /frontier/task/repo /frontier/task/runtime

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
