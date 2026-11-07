# Verifier control image of polygon, both modes: the shared Verifier base plus this task's
# packages (the judge validates every returned counterexample on its own MariaDB here), the
# shared command-sandbox adapter (gpu-tools; the searches run through run_sandboxes() on
# [test_sandboxes.search], whose machine is this image plus the shared CPU runner layer: the
# searches' toolchain, the same pins, and no MariaDB server running there), judge bundle and
# generated config. The platform wrapper invokes /opt/frontier/verifier; task
# evaluator code never receives a Modal, AWS or Kubernetes credential and must not reach S3
# directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG GPU_TOOLS
ARG VERIFIER_BASE
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The transitive pins keep the versions the old python:3.11 image resolved: on Python 3.12
# typing_extensions is no longer pulled in, but the retained parser imports it.
# No apt z3: Ubuntu 24.04 ships Z3 4.8.12, and frontier-build.sh requires 4.13 first on PATH
# (/usr/bin before /usr/local/bin). The z3-solver==4.13.3.0 wheel provides /usr/local/bin/z3,
# the same 4.13.3 the old Debian image had in /usr/bin.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libnss-wrapper mariadb-server openssh-client python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 bidict==0.24.1 lark==1.3.1 mo-sql-parsing==11.699.26234 multiprocess==0.70.19 mysql-connector-python==26.7.0 pandas==3.0.6 prettytable==3.18.0 tqdm==4.70.1 ujson==6.0.0 z3-solver==4.13.3.0 \
       numpy==2.4.6 packaging==26.3 python-dateutil==2.9.0.post0 six==1.17.0 typing_extensions==4.16.0 wcwidth==0.9.2

LABEL org.opencontainers.image.title="FrontierCS2 verifier polygon" \
      dev.frontier.problem-id="polygon"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/fcs2_verifier_gpu.py /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
