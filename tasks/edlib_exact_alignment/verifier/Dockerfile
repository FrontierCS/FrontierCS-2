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
    sh /context/fetch/apt_install.sh build-essential cmake git python3-yaml \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier edlib_exact_alignment" \
      dev.frontier.problem-id="edlib_exact_alignment"

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

# Both judges allow 0.1 s between the program's own timer line and the verifier's elapsed time
# (`evaluator.py:407`, `checked_duration`): isolate's launcher, which moves itself off the
# command's CPU (`isolate.py:1345-1348`), stamps the start when it wakes after the exec, and
# under host load that wake is late, so a short run can print a time longer than the stamped
# one (60 of 500 runs at 30 ms, by 5 ms at worst). The compared time is the verifier's.
#
# The held-out workloads, their correctness seed and the qualification record's final
# reference score are for the final evaluation alone. The train judge's copies of
# `evaluator.py`, `judge/workloads.py`, `judge/workload_manifest.json` and
# `qualification.json` (`train/`) lack them, and are laid over the judge in a stage of its
# own, from which the train judge's image copies the judge, so that they are in none of its
# layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN ! grep --recursive --quiet 'heldout-' /judge && ! grep --quiet '"split": "heldout"' /judge/judge/workload_manifest.json \
    && sed --in-place '407s/^    if reported>elapsed:$/    if reported>elapsed+0.1:/' /judge/evaluator.py \
    && grep --quiet --fixed-strings 'if reported>elapsed+0.1:' /judge/evaluator.py

# The train judge's image: the judge without the held-out workloads, the second correctness
# seed or the qualification record.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge, with the timer allowance above.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
RUN sed --in-place '407s/^    if reported>elapsed:$/    if reported>elapsed+0.1:/' /opt/frontier/task/evaluator.py \
    && grep --quiet --fixed-strings 'if reported>elapsed+0.1:' /opt/frontier/task/evaluator.py
USER 65532:65532
