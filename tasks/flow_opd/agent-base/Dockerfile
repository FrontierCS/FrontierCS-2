# The source's offline runtime (`runtime/Dockerfile`): Debian 12 with Python 3.10, PyTorch 2.6,
# Diffusers, PEFT and PaddleOCR, and SD3.5 Medium and the OCR teacher at /opt/frontier/models. It
# already holds bash, git and tar, so the agent's image installs no package.
FROM runtime

USER root
# The submission tools run the shared agent base's own `/usr/bin/python3`, which this base lacks:
# its Python is the one at /usr/local/bin. The agent runs as uid 1000, user `ubuntu` as in the
# shared agent base, which owns /workspace.
RUN ln --symbolic /usr/local/bin/python3 /usr/bin/python3 \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /workspace --shell /bin/bash ubuntu \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task flow_opd" \
      dev.frontier.problem-id="flow_opd"

# The solver-visible tree, and nothing else: the evaluator, the reference, the workloads and the
# corpora are in the verifier's image only. The submission tools are the repository's `shared/`
# ones, which the build puts in its context, linked from /frontier/task as the shared agent base
# links them.
COPY --chmod=a+rX,a-w shared/agent/bin/fcs2_submission.py /frontier/task/bin/fcs2_submission.py
COPY --chmod=0555 shared/agent/bin/make_submission.sh shared/agent/bin/prepare_workspace.sh /frontier/task/bin/
COPY --chmod=a+rX,a-w shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY --chmod=a+rX,a-w generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --chmod=a+rX,a-w --from=readme /README.md /frontier/task/README.md
COPY --chmod=a+rX,a-w repo /frontier/task/repo
RUN ln --symbolic /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln --symbolic /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && chmod a-w /frontier/task /frontier/task/bin /frontier/task/repo

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
