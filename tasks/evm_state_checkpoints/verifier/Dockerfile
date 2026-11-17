# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS tools

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh autoconf automake bzip2 ca-certificates cmake curl default-libmysqlclient-dev dpkg-dev file g++ gcc git gnupg imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libglib2.0-dev libgmp-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmaxminddb-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt1-dev libyaml-dev make mercurial netbase openssh-client patch procps python3 sq subversion unzip wget xz-utils zlib1g-dev

LABEL org.opencontainers.image.title="FrontierCS2 verifier evm_state_checkpoints" \
      dev.frontier.problem-id="evm_state_checkpoints"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier: the repository's `shared/` ones, which the build puts in its context, and
# which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The generated verifier file and each mode's selector. The starter and the generated config are
# copied below, plain into the stage that builds the seed from them, and read-only into `common`.
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_SPECTRUM_CACHE=/frontier/spectrum-cache FRONTIER_SPECTRUM_CACHE_SEED=/opt/frontier/spectrum-cache-seed
WORKDIR /frontier

# The Spectrum benchmark, built once, at image build time, where the run-scoped writable cache is,
# and
# published as a read-only seed, which the evaluator copies back at run time. It is built in a
# stage of its
# own, from the whole judge, and the images copy only the seed, which holds the starter's build
# and nothing
# of the judge. Its dependencies are cloned from the pinned mirror, and Hunter's downloads are put
# in place
# and the configure is held to them, as in the agent's image.
FROM tools AS spectrum
COPY repo /opt/frontier/task/pristine
COPY generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY judge /opt/frontier/task
RUN --mount=type=bind,source=fetch/git_mirror.sh,target=/context/fetch/git_mirror.sh \
    --mount=type=bind,source=fetch/download.sh,target=/context/fetch/download.sh \
    --mount=type=bind,source=github-pins.txt,target=/context/github-pins.txt \
    --mount=type=bind,source=hunter-pins.txt,target=/context/hunter-pins.txt \
    sh /context/fetch/git_mirror.sh /context/github-pins.txt /opt/frontier-pins \
    && sh /context/fetch/download.sh /context/hunter-pins.txt /frontier/.hunter \
    && FRONTIER_SPECTRUM_CACHE=/frontier/spectrum-cache PYTHONPATH=/opt/frontier/task:/opt/frontier \
       https_proxy=http://127.0.0.1:9 no_proxy=raw.githubusercontent.com \
       python3 -c 'import evaluator; r = evaluator.prepare(); assert r.get("ready"), r' \
    && cp --archive /frontier/spectrum-cache /opt/frontier/spectrum-cache-seed \
    && chmod --recursive a+rX /opt/frontier/spectrum-cache-seed \
    && git config --system --unset-all 'url.file:///opt/frontier-pins/.insteadOf' && rm --recursive --force /opt/frontier-pins

# What both judges' images hold beyond the tools: the starter that the shared verifier overlays
# the submission on, and the judge builds its shipped and reference backends from; the generated
# config; and the seed, so that their layers are in both images.
FROM tools AS common
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
RUN chmod a-w /opt/frontier/task /opt/frontier/task/pristine
COPY --from=spectrum /opt/frontier/spectrum-cache-seed /opt/frontier/spectrum-cache-seed

# The final suite's seeds are for the final evaluation alone. The train judge's `evaluator.py`
# (`train/`) has none, and is laid over the judge in a stage of its own, from which the train
# judge's image copies the judge, so that the seeds are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN echo "d7babb8fd70eccea80b7e93126c00ca7058b9a29ac6a8b682ef9df9ca654cd70  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge with the final suite's seeds left out, and the Spectrum
# benchmark built.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is, and the Spectrum benchmark built.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
