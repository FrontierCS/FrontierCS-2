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
    sh /context/fetch/apt_install.sh build-essential ca-certificates git libtbb-dev python3 python3-pip \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier clusterwise_spgemm" \
      dev.frontier.problem-id="clusterwise_spgemm"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to and builds its baseline kernel from; and the generated files.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite's matrix list, its row of the dataset's member list and its matrix archive are
# for the final evaluation alone. The train judge's `workloads.json` and `dataset-members.json`
# (`train/`) lack them, and are laid over the judge in a stage of its own, from which the train
# judge's image copies the judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN python3 -I -c "import json; rows = json.load(open('/judge/workloads.json'))['final']; assert [r['id'] for r in rows] == ['final-0%d' % i for i in range(8)] and all(set(r) == {'id', 'matrix_sha256'} for r in rows); assert set(json.load(open('/judge/dataset-members.json'))) == {'train'}"

# The train judge's image: the judge without the final suite's matrices, names or archive, and
# with the train archive, which the judge reads at `FRONTIER_DATASET_PATH`.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
COPY datasets/train.tar /opt/frontier/datasets/train.tar
# A checkout without `git lfs` holds a pointer here; the digest is `dataset_objects.train` of
# `generated/judge/config.yaml`.
RUN echo "e7d56702e8a5050bd473adf789d99813dfe0e9d3ee3b9b7b5388530e4148940e  /opt/frontier/datasets/train.tar" | sha256sum --check -
USER 65532:65532

# The final judge's image: the judge as it is, with the final suite's archive.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
COPY datasets/eval.tar /opt/frontier/datasets/eval.tar
# A checkout without `git lfs` holds a pointer here; the digest is `dataset_objects.eval` of
# `generated/judge/config.yaml`.
RUN echo "c93be05b1a9299cb21f346cb6e7afb1c05e30d8f43902d0a8ff1a06a459288b5  /opt/frontier/datasets/eval.tar" | sha256sum --check -
USER 65532:65532
