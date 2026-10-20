# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
LABEL org.opencontainers.image.title="FrontierCS2 task parbutterfly" \
      dev.frontier.problem-id="parbutterfly" \
      dev.frontier.generator="manual-transfer"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 tar \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace \
    && ln -s /opt/frontier/candidate/bin/frontier-submit /usr/local/bin/frontier-submit
COPY agent-base/frontier/task /frontier/task
COPY agent-base/frontier/task/STATEMENT.md /frontier/task/repo/README.md
RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md
USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
