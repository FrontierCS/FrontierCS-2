# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3
# The four graphs are Git LFS files: a checkout without them holds pointer files, and the check
# fails the build then. `workloads.sha256` is the SHA-256 of each, as the pointers give them.
RUN --mount=type=bind,source=judge/workloads,target=/context/judge/workloads \
    --mount=type=bind,source=workloads.sha256,target=/context/workloads.sha256 \
    cd /context && sha256sum --check --strict workloads.sha256

LABEL org.opencontainers.image.title="FrontierCS2 verifier hypergraph_core_numbers" \
      dev.frontier.problem-id="hypergraph_core_numbers"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The graded graphs, and the manifest that gives the parameters they were generated with, are
# for the final evaluation alone. The train judge's copy of the judge is made without them, in
# a stage of its own, from which the train judge's image copies it, so that they are in none
# of its layers.
FROM common AS judge-train
COPY judge /judge
RUN rm --recursive --force /judge/workloads/graded-a.adj /judge/workloads/graded-b.adj /judge/artifact_manifest.json
RUN test ! -e /judge/workloads/graded-a.adj && test ! -e /judge/workloads/graded-b.adj && test ! -e /judge/artifact_manifest.json

# The train judge's image: the judge without the graded graphs.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
