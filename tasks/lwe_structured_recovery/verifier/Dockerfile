# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, the user 65532 with /frontier its home, and the shared verifier's launcher,
# /opt/frontier/verifier.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
LABEL org.opencontainers.image.title="FrontierCS2 verifier lwe_structured_recovery" \
      dev.frontier.problem-id="lwe_structured_recovery"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` one, which the build puts in its context, and
# which replaces the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py /opt/frontier/
# The task's judge, with the starter repo it trusts at pristine/: the checker and catalog are
# read from there, never from the submission. And each mode's selector: the train judge reads
# train.json, and the final judge eval.json.
COPY --chmod=a+rX,a-w judge /opt/frontier/task
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
