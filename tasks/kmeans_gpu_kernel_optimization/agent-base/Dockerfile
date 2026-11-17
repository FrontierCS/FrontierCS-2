# The source's Agent runner base, PyTorch 2.6.0 with CUDA 12.4 and Triton, pinned by digest: the
# toolchain that the task's checks run with.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The submission tools need git and tar: the base holds tar and ca-certificates, so git is the one
# package installed. apt is set to one dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`),
# which replaces the base's archive and NVIDIA's CUDA repository, so that a rebuild installs the
# same version. The agent runs as uid 1000, user `ubuntu` as in the shared agent base, which owns
# /workspace.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task kmeans_gpu_kernel_optimization" \
      dev.frontier.problem-id="kmeans_gpu_kernel_optimization"

# The solver-visible tree, and nothing else: the evaluator, the reference and the final workloads
# are in the verifier's image only. The submission tools are the repository's `shared/` ones,
# which the build puts in its context, linked from /frontier/task as the shared agent base links
# them.
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
