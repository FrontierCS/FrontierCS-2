# The source's Verifier GPU runner image: the runtime the
# agent's image shares (`runtime/Dockerfile`), where the judge runs the warm-up, the candidate and
# the reference under `isolate` on the GPU.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The user 65532 with /frontier its home, as the shared verifier base has them.
RUN install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier detok_latent_denoising" \
      dev.frontier.problem-id="detok_latent_denoising"

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
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite is for the final evaluation alone. `train/trim.py` leaves it out of a copy of
# the judge in a stage of its own, from which the train judge's image copies the judge, so that
# it is in none of the train image's layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN /opt/frontier/runtime-python /trim.py /judge \
    && echo "bd66f733410894085bfbe4332f13e60a86cc89d73216de82d18e97139deaae07  /judge/judge_data/private/assets-manifest.json" | sha256sum --check --strict - \
    && echo "5fe163dba14ab064f4bb7f56c6f71b2d2ed017388bf8ef8866694a81dcdd505a  /judge/judge_data/private/attribution.json" | sha256sum --check --strict - \
    && test "$(ls /judge/judge_data/private | tr '\n' ' ')" = "assets-manifest.json attribution.json development " \
    && ! grep --recursive --quiet --regexp='"suite": "final"' --regexp='"final/' /judge/judge_data

# The train judge's image: the judge with the final suite left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
