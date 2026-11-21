# The source's Verifier GPU runner base, CUDA 12.8.1 devel on Ubuntu 24.04 with the source's Python
# environment and Qwen3-4B, built once for the agent's image and this one (`runtime/Dockerfile`).
# Each judge's image is also its shard machines', where the harness runs on the machine's GPU.
FROM runtime AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive

# The user 65532 with /frontier its home, as the shared verifier base has them.
RUN install --directory --owner=65532 --group=65532 /frontier \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

LABEL org.opencontainers.image.title="FrontierCS2 verifier fast_kv_compaction" \
      dev.frontier.problem-id="fast_kv_compaction"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer. A shard
# machine's commands run in /workspace, as FrontierCS's GPU runner image makes it, owned by the
# machine's user.
RUN ln --symbolic /pumpkin/out /frontier/output \
    && install --directory --owner=65532 --group=65532 /workspace

# The shared verifier: the repository's `shared/` ones, which the build puts in its context; and
# the port's adapter through which the judge leases its shard machines.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     shared/lib/install_frontier_pth.py shared/verifier-lease/fcs2_verifier_gpu.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the patch that the reference
# agent submits (the starter with a comment added: the judge refuses an empty patch); the
# generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w reference.patch /opt/frontier/task/reference.patch
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
# The shared isolate launcher, with which the harness runs the submitted method, is on the judge
# interpreter's default import path (`frontier.pth`: `import isolate`).
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import isolate, zstandard, torch, vllm, transformers' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite's articles are for the final evaluation alone. `train/trim.py` leaves them out
# of a copy of the judge in a stage of its own, from which the train judge's image copies the
# judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN /opt/frontier/runtime-python /trim.py /judge \
    && echo "c665b4ae70145064b5425add9dea8f8aa0292930cb2ae1c2088045a5b16fa022  /judge/evaluator.py" | sha256sum --check --strict - \
    && echo "6d61bf840f1ab1622ea8c29b03f5727c6a3408eac7ccbfdfbfc1bc6f0f5d111b  /judge/data/quality_subset.jsonl" | sha256sum --check --strict - \
    && echo "954175b05974dcda4540066cdb1c3c6017e09e3ae056633cee9a093fc710f4b6  /judge/targets/manifest.json" | sha256sum --check --strict -

# The train judge's image: the judge with the final suite's articles left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
