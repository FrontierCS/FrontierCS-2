# The source's Verifier GPU runner base, CUDA 12.6.3 on Ubuntu 24.04 with Python 3.12 and the
# hash-locked JAX / Brax / MuJoCo MJX environment, built once for the agent's image and this one
# (`runtime/Dockerfile`), where the judge trains the reference and the candidate on the GPU, each in
# an `isolate` instance.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive

# The user 65532 with /frontier its home, as the shared verifier base has them.
RUN install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier mujoco_playground_cpu" \
      dev.frontier.problem-id="mujoco_playground_cpu"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launcher is the shared verifier base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and that the judge admits against
# its manifest and copies for the reference; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import zstandard, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The seeds of the final suite are for the final evaluation alone. The train judge's
# `judge_deps/workloads.json` (`train/`) has none, and is laid over the judge in a stage of its own,
# from which the train judge's image copies the judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/judge_deps/workloads.json /judge/judge_deps/workloads.json
RUN echo "49bd05838143f15427b4ea90948c41b085b5e141bedc25da2db3078bc1c442a5  /judge/judge_deps/workloads.json" | sha256sum --check --strict - \
    && ! grep --recursive --quiet --fixed-strings --regexp='62869' --regexp='"final-' /judge

# The train judge's image: the judge with the final suite left out. The build checks that the
# pristine starter is the one the judge admits (the check the judge makes at the start of a suite).
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
RUN /opt/frontier/runtime-python -B -c 'import json, sys; sys.path.insert(0, "/opt/frontier/task/judge_deps"); import runtime; runtime.admit_pristine("/opt/frontier/task/pristine", runtime.load_source_manifest()); runtime.admit_trusted_children(runtime.load_source_manifest())'
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
RUN /opt/frontier/runtime-python -B -c 'import json, sys; sys.path.insert(0, "/opt/frontier/task/judge_deps"); import runtime; runtime.admit_pristine("/opt/frontier/task/pristine", runtime.load_source_manifest()); runtime.admit_trusted_children(runtime.load_source_manifest())'
USER 65532:65532
