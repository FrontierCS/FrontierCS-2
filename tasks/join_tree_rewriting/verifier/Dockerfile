ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root

LABEL org.opencontainers.image.title="FrontierCS2 verifier join_tree_rewriting" \
      dev.frontier.problem-id="join_tree_rewriting"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge, as the final judge has it: the source's `judge/` without `test_fixtures/`, a
# solution-like patch that nothing reads. The data and the executor are Git LFS files: a
# checkout without them holds pointer files, and the check, against the SHA-256 that the pointers
# give (`lfs.sha256`), fails the build then. The judge is read-only as it is copied, and the
# images copy the result.
FROM common AS judge-eval
COPY --chmod=a+rX,a-w judge /judge
RUN --mount=type=bind,source=lfs.sha256,target=/lfs.sha256 \
    cd /judge && sha256sum --check --strict --quiet /lfs.sha256
RUN rm --recursive --force /judge/test_fixtures

# The final queries' plans are for the final evaluation alone. The train judge's copy of the
# judge is made without them, in a stage of its own, from which the train judge's image copies
# it, so that they are in none of its layers.
FROM judge-eval AS judge-train
RUN rm --recursive --force /judge/judge/workloads/final
RUN test ! -e /judge/judge/workloads/final

# The train judge's image: the judge without the final queries' plans.
FROM common AS train
COPY --from=judge-train /judge /opt/frontier/task
RUN /opt/frontier/task/judge/bin/join_runner --help | grep -q -- --validation-outfile
USER 65532:65532

# The final judge's image: the judge as it is, but for `test_fixtures/`.
FROM common AS eval
COPY --from=judge-eval /judge /opt/frontier/task
RUN /opt/frontier/task/judge/bin/join_runner --help | grep -q -- --validation-outfile
USER 65532:65532
