FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The final's case lists are for the final evaluation alone. The train judge's `evaluator.py`
# (`train/`) has the development lists as literals and no others, and is laid over the judge
# in a stage of its own, from which the train judge's image copies the judge, so that the
# final lists are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN grep --quiet '^DEV_4 = (0, 4, 9, 11)$' /judge/evaluator.py && ! grep --quiet 'FINAL_\|_COMPATIBLE' /judge/evaluator.py

# The train judge's image: the judge without the final's case lists.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
