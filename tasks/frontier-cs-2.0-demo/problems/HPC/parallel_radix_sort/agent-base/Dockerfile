FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgmp10 libmpfr6 libmpc3 zlib1g git util-linux python3 tar && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10001 --create-home --user-group --shell /bin/bash frontier && install -d -o frontier -g frontier /workspace
COPY agent-base/frontier/task /frontier/task
RUN chmod 755 /frontier/task/repo/frontier-cxx /frontier/task/repo/frontier-build.sh /frontier/task/repo/frontier-bench.sh /frontier/task/bin/*.sh && test -f /frontier/task/repo/frontier-gcc7.tar.xz && chmod -R a+rX,a-w /frontier/task
USER 10001:10001
WORKDIR /workspace
ENV TMPDIR=/workspace
