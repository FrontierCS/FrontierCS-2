# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git golang-go python3

LABEL org.opencontainers.image.title="FrontierCS2 verifier bounded_evm_path_prediction" \
      dev.frontier.problem-id="bounded_evm_path_prediction"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge builds its reference and calibration predictors in; the generated files; and each mode's selector.
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

# The Ethereum snapshot holds the blocks of both suites: the train judge replays blocks 14650001
# to 14650200 and the final judge 14650001 to 14650600, and it holds up to 14651000. The train
# judge's copy is rebuilt here without the block records above 14650200, by `truncate/main.go`
# (built in the starter's module with the judge's vendored modules, as the judge builds), which
# fails the build unless the kept keys and values have the SHA-256 below. The train judge's copy
# of the judge is made in a stage of its own, with its snapshot replaced by this one, so that the
# removed blocks are in none of the train image's layers.
FROM common AS truncate
COPY repo /src
COPY judge/judge_data/vendor /src/vendor
COPY truncate/main.go /src/tools/truncate/main.go
RUN cd /src \
    && GOFLAGS=-mod=vendor GOTOOLCHAIN=local GOCACHE=/tmp/go-build go build -buildvcs=false -o /usr/local/bin/truncate ./tools/truncate \
    && rm --recursive --force /tmp/go-build
COPY judge/judge_data/ethereumdata_1465_small /snapshot
RUN mkdir /truncated \
    && truncate /snapshot/copychain /truncated/copychain 14650200 5079e60ed4632a69120a2e5ba814f9e450fa9105c47081b90b6dc0c6490fa71f \
    && cp --archive /snapshot/copychain/ancient /truncated/copychain/ancient

# The judge with this port's evaluator (`eval/evaluator.py`) and the Go files it builds the
# replay from (`eval/seer/`), which score the candidate's predictions replayed in the judge's own
# SeerEVM build where the source scores the report that the candidate's build prints, in the final
# judge alone (the evaluator says why). It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py /judge/evaluator.py
COPY eval/seer /judge/seer
RUN printf '%s  %s\n' \
        7542e66348225ace2101dbb497d443c583ef76084cd38a2ac2f7585a3b5a5ce1 /judge/evaluator.py \
        d356179f0669c5af70024c7839ae976fc5c11f7a3d5c8007a5bee758f454c9c3 /judge/seer/var_table.go \
        eb0c02fd5fe959c858abaedd574ac7e3710813a66d5aa646891386b97b92b997 /judge/seer/record.go \
        a820310b4a865d8d58c314a1d0a7df3e22d9c8856dc6c0bc13a35ea5c2559c4e /judge/seer/replay.go \
    | sha256sum --check --strict -

# The train judge's copy of the judge: this port's judge with the snapshot rebuilt above.
FROM judge-eval AS judge-train
RUN rm --recursive --force /judge/judge_data/ethereumdata_1465_small/copychain
COPY --from=truncate /truncated/copychain /judge/judge_data/ethereumdata_1465_small/copychain

# The train judge's image: this port's judge with the snapshot rebuilt up to block 14650200.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: this port's judge, with the whole snapshot.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
