# Verifier image of parallel_radix_sort, both modes: the shared Verifier base plus this task's
# runtime packages for the pinned GCC 7.5/Cilk Plus toolchain, that toolchain unpacked below /usr
# (isolate instances compile and run with it; they cannot read the task bundle), its judge
# (evaluator, benchmark harnesses, reference patch) and the starter repo the shared verifier
# overlays the submission on. Build context: this task's root, with .frontier-generated/ added
# by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgmp10 libmpc3 libmpfr6 python3 tar util-linux zlib1g zstd \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 verifier parallel_radix_sort" \
      dev.frontier.problem-id="parallel_radix_sort"

COPY repo/frontier-gcc7.tar.xz /tmp/frontier-gcc7.tar.xz
RUN python3 -c "import tarfile; tarfile.open('/tmp/frontier-gcc7.tar.xz').extractall('/usr/local/lib/frontier-gcc7', filter='fully_trusted')" \
    && rm /tmp/frontier-gcc7.tar.xz \
    && chmod -R a+rX,a-w /usr/local/lib/frontier-gcc7

COPY judge /opt/frontier/task
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && chmod 0555 /opt/frontier/task/pristine/frontier-cxx /opt/frontier/task/pristine/frontier-build.sh /opt/frontier/task/pristine/frontier-bench.sh

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
