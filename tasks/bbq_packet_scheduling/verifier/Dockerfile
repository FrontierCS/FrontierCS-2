ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
ENV YOWASP_CACHE_DIR=/opt/yowasp-cache
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages and tools, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bubblewrap build-essential ca-certificates git python-is-python3 python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python3 -c 'import hashlib, io, pathlib, urllib.request, zipfile; data = urllib.request.urlopen("https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip", timeout=120).read(); assert hashlib.sha256(data).hexdigest() == "552799a1d76cd177b9b4cc63a3e77823a3d2a6eb4ec006569288abeff28e1ff8"; binary = zipfile.ZipFile(io.BytesIO(data)).read("sv2v-Linux/sv2v"); assert hashlib.sha256(binary).hexdigest() == "b80a8ac6212aa72a196d9d6129cda35f16dbfa799ec61848c46321f845b6b635"; path = pathlib.Path("/usr/local/bin/sv2v"); path.write_bytes(binary); path.chmod(0o555)' \
    && sv2v --version \
    && yowasp-yosys -V \
    && chmod --recursive a+rX /opt/yowasp-cache

LABEL org.opencontainers.image.title="FrontierCS2 verifier bbq_packet_scheduling" \
      dev.frontier.problem-id="bbq_packet_scheduling"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The judge with this port's evaluator (`eval/evaluator.py`), whose regression gate takes a PASS
# line only with a nonce that the generated RTL cannot print.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py /judge/evaluator.py
RUN echo "3a817fe1f895b9a39163b9a8b02f0dd328b690a007409b8539c26304b565920f  /judge/evaluator.py" | sha256sum --check --strict -

# The held-out cases are for the final evaluation alone. The train judge's `evaluator.py`
# (`train/`, `eval/evaluator.py` without them) is laid over the judge in a stage of its own, from
# which the train judge's image copies the judge, so that they are in none of its layers.
FROM judge-eval AS judge-train
COPY train/ /judge/
RUN echo "1fc6bab3bbae69829b75f25bfe6093ab1b334dd8828015bdc83a3c24dc4d488b  /judge/evaluator.py" | sha256sum --check --strict - \
    && grep --quiet '^HELD = \[\]$' /judge/evaluator.py && ! grep --recursive --quiet 'hidden_' /judge

# The train judge's image: the judge with the held-out cases left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge with this port's evaluator.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
