ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The packages that build CPython, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000,
# the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git libbz2-dev libdb-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev tk-dev uuid-dev xz-utils zlib1g-dev \
    && install --directory --owner=1000 --group=1000 /workspace

# CPython 3.10.20, built from the python.org release by its SHA-256 with the python image's
# configure flags, as `python3` in /usr/local, where the agent had it.
RUN python3 -c 'import urllib.request; urllib.request.urlretrieve("https://www.python.org/ftp/python/3.10.20/Python-3.10.20.tgz", "/tmp/Python-3.10.20.tgz")' \
    && echo "4ff5fd4c5bab803b935019f3e31d7219cebd6f870d00389cea53b88bbe935d1a  /tmp/Python-3.10.20.tgz" | sha256sum --check - \
    && mkdir /tmp/python-build \
    && tar --extract --gzip --file=/tmp/Python-3.10.20.tgz --directory=/tmp/python-build --strip-components=1 \
    && cd /tmp/python-build \
    && ./configure --prefix=/usr/local --enable-loadable-sqlite-extensions --enable-optimizations \
       --enable-option-checking=fatal --enable-shared --with-lto --with-ensurepip \
       LDFLAGS="-Wl,-rpath,/usr/local/lib" \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm --recursive --force /tmp/python-build /tmp/Python-3.10.20.tgz \
    && ln --symbolic python3 /usr/local/bin/python && ln --symbolic pip3 /usr/local/bin/pip \
    && /usr/local/bin/python3 --version

# The judge's measurement runtime (judge/measurement-requirements.txt, the same pinned
# artifacts, each by hash in `requirements.txt`), so that the repository runs locally; the
# agent's sandbox has no network.
COPY agent-base/requirements.txt /tmp/measurement-requirements.txt
RUN grep --extended-regexp '^(pip|setuptools|wheel)==' /tmp/measurement-requirements.txt > /tmp/bootstrap-requirements.txt \
    && /usr/local/bin/python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/bootstrap-requirements.txt \
    && /usr/local/bin/python3 -m pip install --no-cache-dir --no-deps --require-hashes --no-build-isolation \
       --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/measurement-requirements.txt \
    && /usr/local/bin/python3 -m pip check \
    && rm --force /tmp/measurement-requirements.txt /tmp/bootstrap-requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 task diffusion_policy_cpu" \
      dev.frontier.problem-id="diffusion_policy_cpu"

# The solver-visible tree, and nothing else: the evaluator, references, private datasets and
# runtime are in the verifier's image only. The repository's submission tools replace the
# base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
