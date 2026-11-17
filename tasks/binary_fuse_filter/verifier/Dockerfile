# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 python3-pip

LABEL org.opencontainers.image.title="FrontierCS2 verifier binary_fuse_filter" \
      dev.frontier.problem-id="binary_fuse_filter"

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

# The calibration's history (`reference/history/`: earlier measurement sources and
# calibrations, with the final's rows and its seed), which nothing reads, is removed in a
# stage of its own, from which the final judge's image copies the rest, so that it is in none
# of its layers.
FROM common AS judge-eval
COPY judge /judge
RUN rm --recursive --force /judge/reference/history

# The final seed is for the final evaluation alone, and the calibration is bound to the
# measurement's exact sources. The train judge's `measurement.py` (`train/`) has no final
# seed; `rebind.py` then keeps the train workloads in the calibration and binds it to that
# measurement. All of it happens in a stage of its own, from which the train judge's image
# copies the judge, so that the final seed and the final workloads' calibration are in none of
# its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN rm --recursive --force /judge/reference/history
COPY rebind.py /rebind.py
RUN python3 /rebind.py /judge \
    && rm /rebind.py
RUN ! grep --recursive --quiet --fixed-strings FINAL_SEED /judge

# The train judge's image: the judge with no final seed, and the calibration of the train
# workloads alone, re-bound to the measurement.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is, without the calibration's history.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
