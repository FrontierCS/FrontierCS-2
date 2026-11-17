# The public flower_photos archive named in the statement (228 MB), fetched by its SHA-256 as the
# source's recipe checks it, and extracted (ImageFolder
# layout), each in a stage of its own: the image copies the extracted tree with `COPY --link`, so
# that its layer depends on the archive's bytes alone and an edit to any other step of the recipe
# neither fetches nor pushes it again.
FROM scratch AS flower-archive
ADD --link --checksum=sha256:4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz /flower_photos.tgz

FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS flower
COPY --from=flower-archive /flower_photos.tgz /tmp/flower_photos.tgz
RUN install --directory /opt/frontier/public \
    && tar --extract --gzip --file /tmp/flower_photos.tgz --directory /opt/frontier/public --no-same-owner

# The source's Agent GPU runner base, CUDA 12.8.1 on Ubuntu 24.04, pinned by digest: the box the
# task's checks run in, with the toolchain and the public photographs of the source's runner.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS agent

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages for the checks (a compiler, git, tar, Python and pip), from one dated Ubuntu
# archive snapshot (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's CUDA
# repository. Then the source's pip packages, each file pinned by SHA-256 (`pins/requirements.in` is
# the source's list, `pins/requirements.txt` is `uv pip compile --generate-hashes` of it under the
# source's constraints, which `pip` installs under `--require-hashes`). The agent runs as uid 1000,
# the base image's `ubuntu`, which owns /workspace.
COPY pins/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 python3-dev python3-pip tar \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=1000 --group=1000 /workspace

# The public photographs, read-only, in a layer of their own.
COPY --link --chmod=a+rX,a-w --from=flower /opt/frontier/public/flower_photos /opt/frontier/public/flower_photos

LABEL org.opencontainers.image.title="FrontierCS2 task dynamic_tanh" \
      dev.frontier.problem-id="dynamic_tanh"

# The solver-visible tree, and nothing else: the evaluator, the reference and the final photographs
# are in the verifier's image only. The submission tools are the repository's `shared/` ones,
# which the build puts in its context, linked from /frontier/task as the shared agent base links
# them.
COPY --chmod=0555 shared/agent/bin/make_submission.sh shared/agent/bin/prepare_workspace.sh /frontier/task/bin/
COPY --chmod=0444 shared/agent/bin/fcs2_submission.py shared/lib/edit_scope.py generated/agent/submission-policy.json /frontier/task/bin/
COPY --chmod=a+rX,a-w --from=readme /README.md /frontier/task/README.md
COPY --chmod=a+rX,a-w repo /frontier/task/repo
# The starter's two scripts are executable, as the source's recipe makes them.
COPY --chmod=0555 repo/frontier-bench.sh repo/frontier-build.sh /frontier/task/repo/
RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
