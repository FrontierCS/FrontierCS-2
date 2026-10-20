# syntax=docker/dockerfile:1.7
# Task CPU base for bbq_packet_scheduling. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="FrontierCS2 task bbq_packet_scheduling" \
      dev.frontier.problem-id="bbq_packet_scheduling" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="r_top01_scheduling_s2r9_python"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap build-essential ca-certificates git python-is-python3 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 toolz==1.0.0 verilator==5.32.0
# Pinned clock-period toolchain, identical in the Agent and Verifier images:
# Yosys 0.69 as a WebAssembly build from PyPI and the sv2v 0.0.13 release binary.
# The module cache is compiled once here so a run does not pay for it.
ENV YOWASP_CACHE_DIR=/opt/yowasp-cache
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir yowasp-yosys==0.69.0.0.post1233 yowasp-runtime==1.96 wasmtime==47.0.1 platformdirs==4.13.0 click==8.5.0 \
    && python3 -c 'import hashlib, io, pathlib, urllib.request, zipfile; data = urllib.request.urlopen("https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip", timeout=120).read(); assert hashlib.sha256(data).hexdigest() == "552799a1d76cd177b9b4cc63a3e77823a3d2a6eb4ec006569288abeff28e1ff8"; binary = zipfile.ZipFile(io.BytesIO(data)).read("sv2v-Linux/sv2v"); assert hashlib.sha256(binary).hexdigest() == "b80a8ac6212aa72a196d9d6129cda35f16dbfa799ec61848c46321f845b6b635"; path = pathlib.Path("/usr/local/bin/sv2v"); path.write_bytes(binary); path.chmod(0o555)' \
    && sv2v --version \
    && yowasp-yosys -V \
    && chmod -R a+rX /opt/yowasp-cache


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /workspace
