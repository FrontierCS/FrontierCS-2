FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir zstandard==0.23.0 && install -d -o 65532 -g 65532 /frontier
COPY judge /opt/frontier/task
COPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/verify_reference.py /opt/frontier/verify_reference.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier && chmod 0555 /opt/frontier/verifier && find /opt/frontier -type d -exec chmod 0555 {} + && find /opt/frontier -type f -name '*.py' -exec chmod 0444 {} + && chmod -R a+rX,a-w /opt/frontier/task
USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
