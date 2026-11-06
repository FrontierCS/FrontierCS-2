# Verifier image of parallel_dbscan, both modes: the shared Verifier base plus this task's
# compilers, judge (evaluator, exact oracles, reference patch) and the starter repo
# the shared verifier overlays the submission on. Build context: this task's root, with
# .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git tar zstd \
    && rm -rf /var/lib/apt/lists/*

# The old Agent and Verifier images compiled with GCC 12 (Debian 12.2.0); the shared Ubuntu base
# ships GCC 13, which measurably changed one historical submission's timings. GCC 12 is installed
# and linked as `gcc`/`g++`/`cc`/`c++`/`cpp` for the Agent, the Makefile and the judge.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc-12 g++-12 cpp-12 \
    && rm -rf /var/lib/apt/lists/* \
    && for tool in gcc g++ cpp gcov gcc-ar gcc-nm gcc-ranlib; do ln -sf "$tool-12" "/usr/bin/$tool"; done \
    && ln -sf gcc-12 /usr/bin/cc && ln -sf g++-12 /usr/bin/c++ \
    && test "$(g++ -dumpversion | cut -d. -f1)" = 12 && test "$(cc -dumpversion | cut -d. -f1)" = 12 \
    && test "$(c++ -dumpversion | cut -d. -f1)" = 12 && test "$(cpp -dumpversion | cut -d. -f1)" = 12

LABEL org.opencontainers.image.title="FrontierCS2 verifier parallel_dbscan" \
      dev.frontier.problem-id="parallel_dbscan"

COPY judge /opt/frontier/task
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
