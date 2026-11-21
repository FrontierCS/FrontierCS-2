# The source's Verifier GPU runner base, Ubuntu 24.04 pinned by digest, with the source's Python
# environment, where the judge trains and times the candidate and the reference under `isolate`
# on the GPU. The base holds no CA certificates, which the apt snapshot's HTTPS and pip's need
# before `ca-certificates` can be installed, so the bundle of the agent's CUDA base (the same
# Ubuntu release, pinned by digest) is copied in first; the package replaces it.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS certificates

FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852 AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages (a compiler, git, tar, Python and pip) from one dated Ubuntu archive
# snapshot (`fetch/apt_snapshot.sh`) that replaces the base's archive. Then the source's pip
# packages, each file pinned by SHA-256 (`pins/requirements.in` is the source's list,
# `pins/requirements.txt` is `uv pip compile --generate-hashes` of it under the source's
# constraints, which `pip` installs under `--require-hashes`); the user 65532 with /frontier its
# home, as the shared verifier base has them; and /workspace, where the evaluator keeps its scratch
# (the checkpoints of its training runs) by default.
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY pins/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 python3-dev python3-pip tar \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && install --directory --owner=65532 --group=65532 /frontier /workspace \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier dynamic_tanh" \
      dev.frontier.problem-id="dynamic_tanh"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launchers are those of the shared verifier base.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
# The evaluator runs the starter's two scripts directly (`evaluator.py` `_train_command`), so they
# are executable, as the source's recipe makes them.
COPY --chmod=0555 repo/frontier-bench.sh repo/frontier-build.sh /opt/frontier/task/pristine/
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import torch, torchvision, timm, zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine /opt/frontier/datasets

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final workloads' photographs, models, batch sizes and seeds are for the final evaluation
# alone. `train/trim.py` leaves them out of a copy of the judge in a stage of its own, from which
# the train judge's image copies the judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN /opt/frontier/runtime-python /trim.py /judge \
    && echo "40b8284d88f2b13c176df08adf9fd264d1315481951b4d041917787152343c18  /judge/judge_assets/flower_photos/manifest.json" | sha256sum --check --strict - \
    && ! grep --recursive --quiet --regexp=final_ --regexp=91001 /judge/judge_assets/flower_photos/manifest.json /judge/judge_assets/flower_photos/images

# The train judge's image: the judge with the final workloads left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
