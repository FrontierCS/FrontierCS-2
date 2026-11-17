# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image. Go 1.23.12 and the estimator's module
# cache come from the toolchain image (`toolchain/Dockerfile`).
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates curl git python3 tar time
COPY --from=go-toolchain /usr/local/go /usr/local/go
COPY --from=go-toolchain /opt/frontier/go-mod /opt/frontier/go-mod
RUN ln --symbolic /usr/local/go/bin/go /usr/local/bin/go \
    && ln --symbolic /usr/local/go/bin/gofmt /usr/local/bin/gofmt

LABEL org.opencontainers.image.title="FrontierCS2 verifier ckks_composable_noise" \
      dev.frontier.problem-id="ckks_composable_noise"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier, and the launcher through which the judge runs every program it builds
# from a submission: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to, at pristine/; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GOMODCACHE=/opt/frontier/go-mod GOPROXY=off
WORKDIR /frontier

# The frozen precision bands of the 29 final configurations are the final's alone. The train
# judge needs the 12 cells of its development feedback, so `train_judge.py` rewrites a copy of
# the judge, in a stage of its own, from which the train judge's image copies it, so that the
# other bands are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY verifier/train_judge.py /train_judge.py
RUN python3 /train_judge.py /judge

# The train judge's image.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the whole judge.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
