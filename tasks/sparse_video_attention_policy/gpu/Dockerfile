FROM runtime
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
COPY --chmod=0555 task_runtime.py /opt/frontier/task-runtime
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot

COPY --from=agent-base /frontier/task/ /frontier/task/
# Both upstream shell entrypoints invoke "python" through the offline model adapter on PATH.
RUN chmod 0555 /frontier/task/repo/.frontier/python

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_GPU_TYPE=H100 FRONTIER_GPU_SECONDS=18000
WORKDIR /workspace
