FROM python:3.12-slim-bookworm@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl jq tar \
    && rm -rf /var/lib/apt/lists/*

COPY frontier/task /frontier/task

RUN chmod 0555 /frontier/task/check.sh \
    /frontier/task/submit.sh \
    /frontier/task/starter/payload/run.sh \
    && chmod 0444 /frontier/task/README.md \
    /frontier/task/starter/payload/solution.py \
    && useradd --create-home --uid 1000 --shell /bin/sh frontier

USER 1000:1000
ENV HOME=/workspace \
    TMPDIR=/workspace
WORKDIR /workspace
