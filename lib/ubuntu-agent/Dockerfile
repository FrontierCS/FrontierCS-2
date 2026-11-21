FROM frontier-agent-base
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_PACKAGES
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    set -f && sh /context/fetch/apt_install.sh $APT_PACKAGES \
    && install --directory --owner=1000 --group=1000 /workspace

ARG TASK_ID
LABEL org.opencontainers.image.title="FrontierCS2 task ${TASK_ID}" \
      dev.frontier.problem-id="${TASK_ID}"

COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY policy/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

ARG REPO_EXECUTABLE
RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh ${REPO_EXECUTABLE:+"$REPO_EXECUTABLE"}

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
