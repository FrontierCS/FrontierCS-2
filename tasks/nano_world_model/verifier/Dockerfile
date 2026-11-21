# The source's Verifier GPU runner: the PyTorch 2.4.1 / CUDA
# 12.4 runtime with the source's packages and the offline models, built once for the agent's image
# and this one (`runtime/Dockerfile`). Each judge's image is also its trusted machine's, where the
# reference runs and the candidate's latents are decoded; the candidate's machine runs the agent's
# image.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The user 65532 with /frontier its home, as the shared verifier base has them; and /workspace,
# which the source's image makes the `frontier` account's.
RUN install --directory --owner=65532 --group=65532 /frontier /workspace \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier nano_world_model" \
      dev.frontier.problem-id="nano_world_model"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones; and the port's adapter through which the
# judge leases its two machines. /opt/frontier goes on the interpreter's import path
# (frontier.pth), so the judge and every worker it starts import isolate by name; the launchers
# are those of the shared verifier base.
COPY --from=task-tools --chmod=0444 /verifier/*.py /lib/edit_scope.py /lib/isolate.py /lib/install_frontier_pth.py \
    /verifier-lease/fcs2_verifier_gpu.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --chmod=0444 generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import torch, torchvision, lpips, piqa, zstandard, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge, and the suites' public data, which the source builds on the control image
# and this build makes here with the same script, on the GPU
# image's PyTorch.
FROM common AS judge-full
COPY judge /judge
RUN /opt/frontier/runtime-python /judge/public_inputs.py /judge/judge_data /judge/public_data

# The final suite is for the final evaluation alone. This stage removes its manifest, its two
# episodes' frames and its public data from the judge, and the train judge's image copies the
# judge from here, so that they are in none of its layers. `states.pth`, `actions.pth` and
# `seq_lengths.pth` hold every episode in one tensor each and stay.
FROM judge-full AS judge-train
RUN rm --recursive /judge/public_data/final /judge/judge_data/final_subset.json \
       /judge/judge_data/point_maze/obses/episode_315.pth /judge/judge_data/point_maze/obses/episode_565.pth \
    && ! grep --recursive --quiet --regexp='"traj_idx": 565' --regexp='"traj_idx": 315' /judge \
    && test "$(ls /judge/judge_data/point_maze/obses | tr '\n' ' ')" = 'episode_392.pth episode_791.pth '

# The train judge's image: the judge with the final suite's clips left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-full /judge /opt/frontier/task
USER 65532:65532
