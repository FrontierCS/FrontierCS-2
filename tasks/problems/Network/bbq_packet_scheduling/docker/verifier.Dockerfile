# Verifier image of bbq_packet_scheduling, both modes: the shared Verifier base plus this task's
# packages (Verilator, YoWASP Yosys, sv2v, bubblewrap), judge bundle and generated config. The
# platform wrapper invokes /opt/frontier/verifier; task evaluator code never receives a Modal,
# AWS or Kubernetes credential and must not reach S3 directly.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
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

LABEL org.opencontainers.image.title="FrontierCS2 verifier bbq_packet_scheduling" \
      dev.frontier.problem-id="bbq_packet_scheduling"

COPY judge /opt/frontier/task
# The starter the shared verifier overlays the submission on (and the judge applies its
# reference patch to); its source is repo/ itself, not a second copy in the task.
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} +

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
