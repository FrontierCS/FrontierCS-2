# The source's Verifier GPU runner, on the runtime that the
# agent's image shares (`runtime/Dockerfile`): where the judge runs every generation of the
# candidate's attention policy on four H100s, each under the shared `isolate` launcher.
#
# The stage `common` is both judges'; `train` and `eval` add the judge each may read: the graded
# suite's seeds and cached targets are for the final judge alone.
FROM runtime AS common

USER root
# The verifier's pip packages, pinned and installed as in the agent's image, from the
# source's own list and constraints for this image (`pins/verifier-gpu.*`). Then the user 65532
# with /frontier its home, as the shared verifier base has them.
COPY pins/verifier-gpu.requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --require-hashes --no-deps --no-build-isolation \
       --only-binary=:all: --no-binary=flashinfer-python --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python3 -c 'import torch, zstandard; assert torch.__version__.startswith("2.6.0"), torch.__version__' \
    && install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier sparse_video_attention_policy" \
      dev.frontier.problem-id="sparse_video_attention_policy"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier and the launcher each generation runs under: the repository's `shared/`
# ones. /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and
# every worker it starts import isolate by name.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on; the generated files; and each
# mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN ln --symbolic "$(command -v python3)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The graded suite is for the final evaluation alone. `train/trim.py` deletes its cached outputs,
# removes its records from the targets' manifest and sets its seeds in the evaluator to 0, in a
# stage of its own, from which the train judge's image copies the judge, so that they are in none
# of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN /opt/frontier/runtime-python /trim.py /judge \
    && cd /judge \
    && echo "44d66dac807019c0670338dd61930a4e68c0905426c556b134eebea16f49081e  evaluator.py" | sha256sum --check --strict - \
    && echo "a22cf4c78a8649eb20a815fe79eff84e68ccb6704d62b1cfafa55a75d3dbc452  dense_targets/manifest.json" | sha256sum --check --strict - \
    && ! ls dense_targets | grep --quiet graded

# The train judge's image: the judge with the graded suite left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
