# The source's offline runtime (`runtime/Dockerfile`), as the agent's image has it, where the judge
# runs each workload under `isolate` on the GPU. It already holds bash, git, tar and zstandard,
# so the image installs no package.
FROM runtime AS common

USER root
# The user 65532 with /frontier its home, as the shared verifier base has them; the shared
# verifier's tools run the shared base's `/usr/bin/python3`, which this base lacks (its Python is
# the one at /usr/local/bin).
RUN install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier \
    && ln --symbolic /usr/local/bin/python3 /usr/bin/python3

LABEL org.opencontainers.image.title="FrontierCS2 verifier flow_opd" \
      dev.frontier.problem-id="flow_opd"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name.
COPY --from=task-tools --chmod=a+rX,a-w /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge's runtime
# adapter; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --chmod=a+rX,a-w generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import zstandard, pytest, torch, diffusers, paddleocr' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && chmod a-w /opt/frontier/datasets /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite's workloads and the source test corpus that holds their prompts are for the
# final evaluation alone. `train/trim.py` leaves them out of a copy of the judge in a stage of
# its own, from which the train judge's image copies the judge, so that they are in none of its
# layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN python3 -I /trim.py /judge \
    && echo "5f8269853f338992755953b2f8130ad64f131f6326f75aba7c88df1992897835  /judge/judge/suite_manifest.json" | sha256sum --check --strict - \
    && echo "ba50d23eec8454fe8c134eb5ad1ff391eb4afcb24616ab6c5d0d6bcce5ae0385  /judge/judge/reference_cache.json" | sha256sum --check --strict - \
    && echo "f797b168544307548401cba018047b12076c8c9ac54b19ed012234abc721d5c9  /judge/judge/qualification.json" | sha256sum --check --strict - \
    && test ! -e /judge/judge/corpus/test.txt \
    && ! grep --recursive --quiet 'final-0' /judge

# The train judge's image: the judge with the final suite left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
