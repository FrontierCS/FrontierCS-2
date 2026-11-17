# The source's Agent GPU runner and its CPU image in one: the runtime that the verifier's image
# shares (`runtime/Dockerfile`: the PyTorch base, the packages, the models), the runner's pip
# packages, and the solver-visible tree. It is also the agent's GPU runner's image, so the agent's
# box keeps the runner's toolchain and models; it drops the runner's SSH daemon and bootstrap.
FROM runtime AS agent
USER root

# The runner's pip packages, each file pinned by SHA-256: `pins/agent-gpu.requirements.txt`
# is `uv pip compile` of the source's list under the source's constraints (`pins/`),
# which `pip` installs under `--require-hashes` without resolving again; the base's
# packages of the same versions stay as they are. `flashinfer-python` has an sdist alone, which is
# built with the environment's own setuptools and no build isolation. The conda Python of the base
# is the one on the path, as for the source's `pip3`. The agent runs as uid 1000, user `ubuntu` as
# in the shared agent base, which owns /workspace.
COPY pins/agent-gpu.requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --require-hashes --no-deps --no-build-isolation \
       --only-binary=:all: --no-binary=flashinfer-python --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python3 -c 'import torch, cmake; assert torch.__version__.startswith("2.6.0"), torch.__version__' \
    && ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task sparse_video_attention_policy" \
      dev.frontier.problem-id="sparse_video_attention_policy"

# The wrapper that runs the statement's commands in the toolchain's environment, with the
# environment it records now.
COPY --chmod=0555 gpu/task_runtime.py /opt/frontier/task-runtime
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot

# The solver-visible tree, and nothing else: the evaluator, the graded suite and its cached
# targets are in the verifier's image only. The submission tools are the repository's `shared/`
# ones, which the build puts in its context, linked from /frontier/task as the shared agent base
# links them.
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
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100 FRONTIER_GPU_SECONDS=18000
WORKDIR /workspace
