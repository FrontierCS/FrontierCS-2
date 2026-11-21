ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's Python packages, as the source pins them, with the packages they need, each file pinned
# by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip compile
# --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh bash ca-certificates git libseccomp2 python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier lejepa_world_model" \
      dev.frontier.problem-id="lejepa_world_model"

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

# The judge with this port's evaluator (`eval/evaluator.py`), which recomputes the quality
# statistics in its own process from the encoder that the training run saved, where the source
# reads them from the report that the training run writes. It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py /judge/evaluator.py
RUN echo "6b776fcb28b190fd2edc127694d32bfaa4ef0ee39ec051878012621e0d967b77  /judge/evaluator.py" | sha256sum --check --strict -

# The held-out seeds are for the final evaluation alone. The train judge's evaluator is the
# final's with `FINAL_SEEDS` emptied, in a stage of its own, from which the train judge's image
# copies the judge, so that the seeds are in none of its layers.
FROM judge-eval AS judge-train
RUN sed --in-place 's/^FINAL_SEEDS = .*$/FINAL_SEEDS = ()/' /judge/evaluator.py \
    && grep --quiet '^FINAL_SEEDS = ()$' /judge/evaluator.py \
    && echo "78c2b3f0f64a51bc294e023634274ddc96f7559433d31985a45c4ca27de31f68  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge with the held-out seeds left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge with this port's evaluator.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
