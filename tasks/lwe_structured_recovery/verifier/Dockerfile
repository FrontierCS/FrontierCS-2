ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
LABEL org.opencontainers.image.title="FrontierCS2 verifier lwe_structured_recovery" \
      dev.frontier.problem-id="lwe_structured_recovery"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /opt/frontier/
# The task's judge, with the starter repo it trusts at pristine/: the checker and catalog are
# read from there, never from the submission. And each mode's selector: the train judge reads
# train.json, and the final judge eval.json.
COPY --chmod=a+rX,a-w judge /opt/frontier/task
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
