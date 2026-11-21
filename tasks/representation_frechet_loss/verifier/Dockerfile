# The source's Verifier GPU runner base, CUDA 12.4.1 with cuDNN on Ubuntu 22.04 with the source's
# Python environment and the offline models, built once for the agent's image and this one
# (`runtime/Dockerfile`), where the judge runs the paired workloads on its GPU and starts every
# submitted queue in an `isolate` instance.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The user 65532 with /frontier its home, as the shared verifier base has them; and /workspace,
# which the source's image makes the `frontier` account's.
RUN install --directory --owner=65532 --group=65532 /frontier /workspace \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier representation_frechet_loss" \
      dev.frontier.problem-id="representation_frechet_loss"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launchers are those of the shared verifier base.
COPY --from=task-tools --chmod=0444 /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --chmod=0444 generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import torch, torchvision, timm, zstandard, scipy, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final workloads' seeds are for the final evaluation alone. The train judge's
# `judge_deps/workloads.json` (`train/`) leaves them out and is laid over the judge in a stage of
# its own, from which the train judge's image copies the judge, so that they are in none of its
# layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN echo "0d14aad785f42c3317c5352fb036bb817fb5b067c98bff2bd39295af409f957d  /judge/judge_deps/workloads.json" | sha256sum --check --strict - \
    && ! grep --recursive --quiet --extended-regexp '62869|83423|103997|73553|94109|114691|final-[0-9]' /judge

# The train judge's image: the judge with the final workloads left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
