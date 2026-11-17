# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's Python packages, each file pinned by SHA-256: `requirements.txt` is `uv pip compile
# --generate-hashes` of the source's list in `requirements.in`, which `pip` installs under
# `--require-hashes`.
COPY verifier/requirements.txt /tmp/requirements.txt
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3 python3-pip util-linux zlib1g-dev \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

LABEL org.opencontainers.image.title="FrontierCS2 verifier runlts_predictor" \
      dev.frontier.problem-id="runlts_predictor"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
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

# The private traces, about 15 GB, are the source's own image, pinned by digest: its
# /opt/runlts-private-data (the 105 traces, byte for byte, a+rX, a-w).
FROM docker.io/joyemang33/fcs2-task-verifier-runlts_predictor@sha256:74b8ba63764bae367ee21dd4d688ddc7b9dfac66e64a7735c920c03e2b825ff7 AS traces

# The judge with this port's evaluator (`eval/evaluator.py`), which scores a replay of the
# predictor's directions where the source scores the counters the candidate's own simulator prints,
# in the final judge alone, and without the files that nothing reads: the lists of where the training traces
# came from and the two patches that only illustrate a solution. Each stage makes its copy of the
# judge in a layer of its own, so that what it removes is in none of the images' layers.
FROM common AS judge-eval
COPY --chmod=a+rX,a-w judge /judge
COPY --chmod=a+rX,a-w eval/evaluator.py /judge/evaluator.py
RUN echo "6a5785cb9cb6c12b70c9a029cbcf3a17a680b4d56f88fed54f004c9dc142a5cf  /judge/evaluator.py" | sha256sum --check --strict - \
    && rm --recursive --force \
    /judge/training_sources.json \
    /judge/training_pins.json \
    /judge/partial_example.patch \
    /judge/baseline_naive.patch

# The train judge's judge: the same, with the table of the reference's counters and the
# evaluator rewritten for the 24 development traces (`train/derive.py` says
# how), which are all the traces its image holds. The script fails unless its results have the
# digests below.
FROM judge-eval AS judge-train
RUN --mount=type=bind,source=train/derive.py,target=/derive.py \
    python3 -I /derive.py /judge 077f93146bbb925266f1edec43e916a326845148b1f1aab20d4bf95bfc127628 \
    47826984e48bdfee667b75dd01e03d0b5694b738db6ee67e4247edeee9b26116

# The train judge's image: the judge of the train stage, and the 24 development traces (3.7
# GB) alone, in the directories of the full set.
FROM common AS train
COPY --from=judge-train /judge /opt/frontier/task
COPY --link --from=traces /opt/runlts-private-data/traces/compress/compress_1_trace.gz \
     /opt/runlts-private-data/traces/compress/compress_7_trace.gz \
     /opt/runlts-private-data/traces/compress/readme.txt \
     /opt/runlts-private-data/traces/compress/
COPY --link --from=traces /opt/runlts-private-data/traces/fp/fp_10_trace.gz \
     /opt/runlts-private-data/traces/fp/fp_13_trace.gz \
     /opt/runlts-private-data/traces/fp/fp_9_trace.gz \
     /opt/runlts-private-data/traces/fp/readme.txt \
     /opt/runlts-private-data/traces/fp/
COPY --link --from=traces /opt/runlts-private-data/traces/infra/infra_12_trace.gz \
     /opt/runlts-private-data/traces/infra/infra_2_trace.gz \
     /opt/runlts-private-data/traces/infra/infra_4_trace.gz \
     /opt/runlts-private-data/traces/infra/infra_5_trace.gz \
     /opt/runlts-private-data/traces/infra/readme.txt \
     /opt/runlts-private-data/traces/infra/
COPY --link --from=traces /opt/runlts-private-data/traces/int/int_12_trace.gz \
     /opt/runlts-private-data/traces/int/int_18_trace.gz \
     /opt/runlts-private-data/traces/int/int_22_trace.gz \
     /opt/runlts-private-data/traces/int/int_31_trace.gz \
     /opt/runlts-private-data/traces/int/int_32_trace.gz \
     /opt/runlts-private-data/traces/int/int_36_trace.gz \
     /opt/runlts-private-data/traces/int/int_5_trace.gz \
     /opt/runlts-private-data/traces/int/int_7_trace.gz \
     /opt/runlts-private-data/traces/int/readme.txt \
     /opt/runlts-private-data/traces/int/
COPY --link --from=traces /opt/runlts-private-data/traces/media/media_2_trace.gz \
     /opt/runlts-private-data/traces/media/readme.txt \
     /opt/runlts-private-data/traces/media/
COPY --link --from=traces /opt/runlts-private-data/traces/web/web_0_trace.gz \
     /opt/runlts-private-data/traces/web/web_12_trace.gz \
     /opt/runlts-private-data/traces/web/web_13_trace.gz \
     /opt/runlts-private-data/traces/web/web_16_trace.gz \
     /opt/runlts-private-data/traces/web/web_20_trace.gz \
     /opt/runlts-private-data/traces/web/web_6_trace.gz \
     /opt/runlts-private-data/traces/web/readme.txt \
     /opt/runlts-private-data/traces/web/
RUN ln --symbolic /opt/runlts-private-data /opt/frontier/task/private_data \
    && chmod 0555 /opt/runlts-private-data /opt/runlts-private-data/traces \
    /opt/runlts-private-data/traces/compress \
    /opt/runlts-private-data/traces/fp \
    /opt/runlts-private-data/traces/infra \
    /opt/runlts-private-data/traces/int \
    /opt/runlts-private-data/traces/media \
    /opt/runlts-private-data/traces/web
USER 65532:65532

# The final judge's image: the judge as it is (without those files), and all 105 traces.
FROM common AS eval
COPY --from=judge-eval /judge /opt/frontier/task
COPY --link --from=traces /opt/runlts-private-data /opt/runlts-private-data
RUN ln --symbolic /opt/runlts-private-data /opt/frontier/task/private_data \
    && chmod 0555 /opt/runlts-private-data
USER 65532:65532
