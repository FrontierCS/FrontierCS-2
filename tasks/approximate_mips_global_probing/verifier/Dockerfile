# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential ca-certificates git python3

LABEL org.opencontainers.image.title="FrontierCS2 verifier approximate_mips_global_probing" \
      dev.frontier.problem-id="approximate_mips_global_probing"

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

# The judge with this port's evaluator (`eval/evaluator.py`), which recomputes the inner products
# of the printed results with the judge's own program (`eval/inner_products.cpp`), where the
# source's judge takes the instance's. It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py eval/inner_products.cpp /judge/
RUN echo "3130c1d46846d5942c2597cc00ee82493e0bb3fab04f4e70b82961f9c273fbd0  /judge/evaluator.py" | sha256sum --check --strict - \
    && echo "7c73d0434033159cf1ac2707a993e3ba646ee52f200390a0e6218d6c15775ce9  /judge/inner_products.cpp" | sha256sum --check --strict -

# The second split's hash seed and the ground truth of its queries are for the final
# evaluation alone. The train judge's `evaluator.py` and ground-truth files (`train/`) lack
# them, and are laid over the judge in a stage of its own, from which the train judge's image
# copies the judge, so that they are in none of its layers.
FROM judge-eval AS judge-train
COPY train/ /judge/
RUN sed -i 's/^    seed = HASH_SEED if split == 0 else .*$/    seed = HASH_SEED/' /judge/evaluator.py \
    && grep --quiet '^    seed = HASH_SEED$' /judge/evaluator.py

# The train judge's image: the judge without the second split's seed or the ground truth of
# its queries.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge with this port's evaluator.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
