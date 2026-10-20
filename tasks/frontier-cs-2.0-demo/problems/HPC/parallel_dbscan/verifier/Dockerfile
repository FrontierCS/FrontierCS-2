FROM python:3.11-bookworm@sha256:39ad9558cfd5729109d27accfeba9f8b019e5d60cb35b3d0a1404a2357a3661e
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git tar zstd && rm -rf /var/lib/apt/lists/*
COPY judge /opt/frontier/judge/
COPY verifier/verify.py /opt/frontier/verifier/verify.py
COPY verifier/beat_spec_feedback.py /opt/frontier/verifier/beat_spec_feedback.py
RUN chmod -R a=rX /opt/frontier
ENV PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
WORKDIR /frontier
