# The source's Verifier GPU runner: the veRL / vLLM image with
# the source's packages and the offline model, built once for the agent's image and this one
# (`runtime/Dockerfile`). The judge runs the candidate's PPO run and the reference's at the same
# time on two machines it leases, of the agent's image.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The user 65532 with /frontier its home, as the shared verifier base has them; and /workspace,
# which the source's image makes the `frontier` account's.
RUN install --directory --owner=65532 --group=65532 /frontier /workspace \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier ragen_multi_turn_rl" \
      dev.frontier.problem-id="ragen_multi_turn_rl"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context; and
# the port's adapter through which the judge leases its machines. /opt/frontier goes on the
# interpreter's import path (frontier.pth), so the judge imports isolate by name.
COPY --chmod=0444 shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     shared/lib/install_frontier_pth.py shared/verifier-lease/fcs2_verifier_gpu.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --chmod=0444 generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import torch, vllm, zstandard, yaml, gym_sokoban' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge, as the source has it.
FROM common AS judge-eval
COPY judge /judge

# The final suite's seed and group selection are for the final evaluation alone, two lines of the
# evaluator. The train judge's image copies the judge from this stage, in which each is made the
# development value that a train call uses, so that neither is in any of its layers.
FROM judge-eval AS judge-train
RUN sed --in-place \
        --expression 's/seed = 314159 if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else 271828/seed = 271828/' \
        --expression 's/selected = \[1, 3, 5\] if final else \[0, 2, 4\]/selected = [0, 2, 4]/' /judge/evaluator.py \
    && grep --quiet --fixed-strings --line-regexp '        seed = 271828' /judge/evaluator.py \
    && grep --quiet --fixed-strings --line-regexp '        selected = [0, 2, 4]' /judge/evaluator.py \
    && ! grep --quiet --extended-regexp '314159|1, 3, 5' /judge/evaluator.py

# The train judge's image: the judge with the final suite's seed and groups left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
