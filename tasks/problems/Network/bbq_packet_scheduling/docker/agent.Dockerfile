# Agent image of bbq_packet_scheduling: the shared Agent base plus this task's packages
# (Verilator, YoWASP Yosys, sv2v, bubblewrap), statement, starter repo and generated submission
# policy. Composed with the Candidate overlay at Run time, so it must not depend on its own
# ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap build-essential ca-certificates git python-is-python3 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir PyYAML==6.0.3 toolz==1.0.0 verilator==5.32.0
ENV YOWASP_CACHE_DIR=/opt/yowasp-cache
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir yowasp-yosys==0.69.0.0.post1233 yowasp-runtime==1.96 wasmtime==47.0.1 platformdirs==4.13.0 click==8.5.0 \
    && python3 -c 'import hashlib, io, pathlib, urllib.request, zipfile; data = urllib.request.urlopen("https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip", timeout=120).read(); assert hashlib.sha256(data).hexdigest() == "552799a1d76cd177b9b4cc63a3e77823a3d2a6eb4ec006569288abeff28e1ff8"; binary = zipfile.ZipFile(io.BytesIO(data)).read("sv2v-Linux/sv2v"); assert hashlib.sha256(binary).hexdigest() == "b80a8ac6212aa72a196d9d6129cda35f16dbfa799ec61848c46321f845b6b635"; path = pathlib.Path("/usr/local/bin/sv2v"); path.write_bytes(binary); path.chmod(0o555)' \
    && sv2v --version \
    && yowasp-yosys -V \
    && chmod -R a+rX /opt/yowasp-cache

LABEL org.opencontainers.image.title="FrontierCS2 task bbq_packet_scheduling" \
      dev.frontier.problem-id="bbq_packet_scheduling"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the Verifier image only.
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
