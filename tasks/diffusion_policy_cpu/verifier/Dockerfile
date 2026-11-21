ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# PyYAML, and for the measurement interpreter below PyYAML and zstandard, each file pinned by
# SHA-256 (`requirements.in` and `requirements-3.10.in` are the source's pins; each `.txt` is `uv
# pip compile --generate-hashes` of its `.in` for that interpreter, which `pip` installs under
# `--require-hashes`).
COPY verifier/requirements.txt /tmp/requirements.txt
COPY verifier/requirements-3.10.txt /tmp/requirements-3.10.txt
# The judge's packages, from the same Ubuntu 24.04 snapshot.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bubblewrap ca-certificates git libgl1 libglib2.0-0 libbz2-1.0 libcrypt1 libdb5.3t64 libexpat1 libffi8 libgdbm6t64 libgssapi-krb5-2 liblzma5 libncursesw6 libnsl2 libreadline8t64 libsqlite3-0 libssl3t64 libtirpc3t64 libuuid1 zlib1g python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The release's interpreter, byte for byte (python:3.10.20-slim-bookworm), relocated.
COPY --from=docker.io/library/python@sha256:6dafe336b64dbf76f7ec872d07e58b5c7de0e2ea5d4c0853a05c3b6be3278b00 /usr/local /opt/python3.10
RUN echo /opt/python3.10/lib > /etc/ld.so.conf.d/python3.10.conf && ldconfig \
    && /opt/python3.10/bin/python3.10 -c 'import ssl, sqlite3, ctypes, lzma, bz2, readline, uuid; print("python3.10 ok")' \
    && /opt/python3.10/bin/python3.10 -m pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements-3.10.txt \
    && rm /tmp/requirements-3.10.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier diffusion_policy_cpu" \
      dev.frontier.problem-id="diffusion_policy_cpu"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
COPY --chmod=a+rX,a-w judge /opt/frontier/task
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; and the generated files. Each mode's selector is the image's own, below.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
# The measurement runtime, built from the judge's pins: /opt/diffusion-policy/python, from the
# wheels that the judge holds and those it fetches by hash, and the checkpoint.
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine \
    && PYTHONDONTWRITEBYTECODE=1 /opt/python3.10/bin/python3.10 /opt/frontier/task/prepare_runtime.py \
    && chmod 0555 /opt/frontier

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite's private dataset is in `datasets/eval.json` alone (its inner archive holds
# the final suite and its sealed constants), so that the train judge's image has only
# `datasets/train.json`, which holds the hashes of the final suite and none of its content.
FROM common AS train
COPY --chmod=a+rX,a-w datasets/train.json /opt/frontier/datasets/train.json
RUN chmod a-w /opt/frontier/datasets
USER 65532:65532

FROM common AS eval
COPY --chmod=a+rX,a-w datasets/eval.json /opt/frontier/datasets/eval.json
RUN chmod a-w /opt/frontier/datasets
USER 65532:65532
