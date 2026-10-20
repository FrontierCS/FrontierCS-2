FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgmp10 libmpfr6 libmpc3 zlib1g git util-linux python3 tar zstd && rm -rf /var/lib/apt/lists/*
COPY agent-base/frontier/task/repo /opt/frontier/repo
COPY judge /opt/frontier/judge
COPY verifier/verify.py /opt/frontier/verifier/verify.py
COPY verifier/beat_spec_feedback.py /opt/frontier/verifier/beat_spec_feedback.py
RUN chmod 755 /opt/frontier/repo/frontier-cxx && test -f /opt/frontier/repo/frontier-gcc7.tar.xz && chmod -R a+rX,a-w /opt/frontier
USER 65532:65532
ENV TMPDIR=/frontier
WORKDIR /frontier
