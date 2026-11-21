FROM frontier-verifier-base

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_PACKAGES
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    set -f && sh /context/fetch/apt_install.sh $APT_PACKAGES

ARG TASK_ID
LABEL org.opencontainers.image.title="FrontierCS2 verifier ${TASK_ID}" \
      dev.frontier.problem-id="${TASK_ID}"
RUN ln --symbolic /pumpkin/out /frontier/output

COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w judge-config/config.yaml /opt/frontier/task/config.yaml
COPY verifier-config/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
