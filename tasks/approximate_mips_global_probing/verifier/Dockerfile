FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

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
