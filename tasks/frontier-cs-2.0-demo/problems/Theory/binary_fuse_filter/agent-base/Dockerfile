FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 --user-group frontier && install -d -o 10001 -g 10001 /workspace
COPY agent-base/frontier/task /frontier/task
RUN find /frontier/task -type d -exec chmod 0555 {} + && find /frontier/task -type f -exec chmod 0444 {} + && find /frontier/task -type f -name '*.sh' -exec chmod 0555 {} +
USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
