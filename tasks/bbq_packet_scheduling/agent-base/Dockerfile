# FrontierCS's shared agent base, pinned by digest: Ubuntu 24.04 with git and Python, and apt
# set to one dated archive snapshot, so that a rebuild installs the same versions, and to try
# each file up to 10 times.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
ENV YOWASP_CACHE_DIR=/opt/yowasp-cache
COPY agent-base/requirements.txt /tmp/requirements.txt
# The task's packages, from the same Ubuntu 24.04 snapshot, then its Python packages, the `sv2v`
# release the source fetches (its archive and its binary each checked against a SHA-256) and the
# Yosys cache warmed to /opt/yowasp-cache. The agent runs as uid 1000, the base image's `ubuntu`,
# which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bubblewrap build-essential ca-certificates git python-is-python3 python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python3 -c 'import hashlib, io, pathlib, urllib.request, zipfile; data = urllib.request.urlopen("https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip", timeout=120).read(); assert hashlib.sha256(data).hexdigest() == "552799a1d76cd177b9b4cc63a3e77823a3d2a6eb4ec006569288abeff28e1ff8"; binary = zipfile.ZipFile(io.BytesIO(data)).read("sv2v-Linux/sv2v"); assert hashlib.sha256(binary).hexdigest() == "b80a8ac6212aa72a196d9d6129cda35f16dbfa799ec61848c46321f845b6b635"; path = pathlib.Path("/usr/local/bin/sv2v"); path.write_bytes(binary); path.chmod(0o555)' \
    && sv2v --version \
    && yowasp-yosys -V \
    && chmod --recursive a+rX /opt/yowasp-cache \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task bbq_packet_scheduling" \
      dev.frontier.problem-id="bbq_packet_scheduling"

# The solver-visible tree, and nothing else: the evaluator, reference and held-out cases are
# in the verifier's image only. The submission tools are the repository's `shared/` ones,
# which the build puts in its context, and which replace the base's.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
