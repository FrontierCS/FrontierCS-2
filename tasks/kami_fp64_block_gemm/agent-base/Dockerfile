# The source's GPU runner base, CUDA 12.8.0 devel on Ubuntu 24.04, pinned by digest: the
# toolchain that the task's checks compile with.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The submission tools need git, tar and python3, and the checks python3: the base holds tar and
# ca-certificates, so git and python3 are installed, from one dated Ubuntu archive snapshot
# (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's CUDA repository. The
# agent runs as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh git python3 \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task kami_fp64_block_gemm" \
      dev.frontier.problem-id="kami_fp64_block_gemm"

# The solver-visible tree, and nothing else: the evaluator, the reference and the hidden
# workloads are in the verifier's image only. The submission tools are linked from
# /frontier/task as in the shared agent base.
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
