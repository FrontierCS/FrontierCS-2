# Agent image of polygon: the shared Agent base plus this task's packages, statement, starter
# repo and generated submission policy. Composed with the Candidate overlay at Run time, so it
# must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The transitive pins keep the versions the old python:3.11 image resolved: on Python 3.12
# typing_extensions is no longer pulled in, but the retained parser imports it.
# No apt z3: Ubuntu 24.04 ships Z3 4.8.12, and frontier-build.sh requires 4.13 first on PATH
# (/usr/bin before /usr/local/bin). The z3-solver==4.13.3.0 wheel provides /usr/local/bin/z3,
# the same 4.13.3 the old Debian image had in /usr/bin.
RUN apt-get update \
    && apt-get install -y --no-install-recommends autoconf automake binutils build-essential ca-certificates curl file git libtool m4 make mariadb-server openssh-client patch pkg-config python-is-python3 python3 python3-dev python3-pip unzip wget \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 bidict==0.24.1 lark==1.3.1 mo-sql-parsing==11.699.26234 multiprocess==0.70.19 mysql-connector-python==26.7.0 pandas==3.0.6 prettytable==3.18.0 tqdm==4.70.1 ujson==6.0.0 z3-solver==4.13.3.0 \
       numpy==2.4.6 packaging==26.3 python-dateutil==2.9.0.post0 six==1.17.0 typing_extensions==4.16.0 wcwidth==0.9.2 contourpy==1.3.3 \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PySMT==0.9.6 iteration-utilities==0.13.0 matplotlib==3.11.2 psycopg2-binary==2.9.13

LABEL org.opencontainers.image.title="FrontierCS2 task polygon" \
      dev.frontier.problem-id="polygon"

# The solver-visible tree, and nothing else: the evaluator, the graded workloads, the reference
# and the beat-reference specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
