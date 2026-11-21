ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The simulator's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages, from the same Ubuntu 24.04 snapshot, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 python3-pip python3-venv \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier shockwave_dynamic_gpu_allocation" \
      dev.frontier.problem-id="shockwave_dynamic_gpu_allocation"

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

# The judge with this port's evaluator (`eval/evaluator.py`), which scores a replay of the
# submitted policy's rounds where the source scores the statistics that policy's own simulator
# computes. It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py /judge/evaluator.py
RUN echo "3e9a7275f3bd02e1a40217433bd70a21802c77e2a3dcd7b15c4c5c6d504e192c  /judge/evaluator.py" | sha256sum --check --strict -

# The final suite's two traces are for the final evaluation alone: the train judge's role reads
# the development suite alone. They are removed from a
# copy of the judge in a stage of its own, from which the train judge's image copies the judge,
# so that they are in none of its layers.
FROM judge-eval AS judge-train
RUN rm /judge/judge_data/final_a.trace /judge/judge_data/final_b.trace \
    && ! find /judge -name 'final_*' | grep --quiet .

# The train judge's image: the judge with the final suite's traces left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge with this port's evaluator, and every trace.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
