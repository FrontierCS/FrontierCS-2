FROM python:3.11-bookworm@sha256:39ad9558cfd5729109d27accfeba9f8b019e5d60cb35b3d0a1404a2357a3661e
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git tar && rm -rf /var/lib/apt/lists/*
COPY agent-base/frontier/task /frontier/task/
RUN chmod -R a=rX /frontier/task && chmod 555 /frontier/task/bin/*.sh /frontier/task/bin/*.py /frontier/task/repo/make_submission.sh
ENV PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
WORKDIR /workspace
