FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The judge with this port's evaluator (`eval/evaluator.py`), which recomputes the inner products
# of the printed results with the judge's own program (`eval/inner_products.cpp`), where the
# source's judge takes the instance's. It is the judge of both modes.
FROM common AS judge-eval
COPY judge /judge
COPY eval/evaluator.py eval/inner_products.cpp /judge/
RUN echo "52709d8c5e920af64bfba11aed208f472328d475dacdc42d7a1f4c53ef232fce  /judge/evaluator.py" | sha256sum --check --strict - \
    && echo "7c73d0434033159cf1ac2707a993e3ba646ee52f200390a0e6218d6c15775ce9  /judge/inner_products.cpp" | sha256sum --check --strict -

# The heap capacities that only the final evaluation measures, and the ground truth of the
# second split's queries, are for the final evaluation alone. The train judge's `evaluator.py`
# and ground-truth file (`train/`) lack them, and are laid over the judge in a stage of its
# own, from which the train judge's image copies the judge, so that they are in none of its
# layers.
FROM judge-eval AS judge-train
COPY train/ /judge/
RUN sed -i 's/^FINAL_ONLY_CAPACITIES = (.\+)$/FINAL_ONLY_CAPACITIES = ()/' /judge/evaluator.py \
    && grep --quiet '^FINAL_ONLY_CAPACITIES = ()$' /judge/evaluator.py

# The train judge's image: the judge without the final-only capacities or the ground truth of
# the second split's queries.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge with this port's evaluator.
FROM common AS eval
COPY --chmod=a+rX,a-w --from=judge-eval /judge /opt/frontier/task
USER 65532:65532
