# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The task's Python packages, as the source pins them, with the packages they need, each file
# pinned by SHA-256 (`requirements.in` is the source's list; `requirements.txt` is `uv pip
# compile --generate-hashes` of it, which `pip` installs under `--require-hashes`).
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git mariadb-server python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier polygon" \
      dev.frontier.problem-id="polygon"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer. Each judge's
# image is also its search machine's, whose commands run in /workspace, as FrontierCS's CPU runner
# layer makes it, owned by the machine's user.
RUN ln --symbolic /pumpkin/out /frontier/output \
    && install --directory --owner=65532 --group=65532 /workspace

# The shared verifier, and the launcher through which a judge runs what it builds from a
# submission: the repository's `shared/` ones, which the build puts in its context, and which
# replace the base's; and the port's adapter through which the judge leases its search machine.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     shared/verifier-lease/fcs2_verifier_gpu.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector of its workloads.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final workloads are removed in a stage of their own, from which the train judge's image
# copies the rest, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
RUN rm /judge/workloads/final.json

# The train judge's image: the task's judge without the final workloads.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the whole judge.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
