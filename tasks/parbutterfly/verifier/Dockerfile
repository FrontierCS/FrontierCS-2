FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The hidden graphs are for the final evaluation alone. The train judge's `evaluator.py`
# (`train/`) has them empty, and is laid over the judge in a stage of its own, from which the
# train judge's image copies the judge, so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN grep --quiet '^HIDDEN = \[\]$' /judge/evaluator.py

# The train judge's image: the judge without the hidden graphs.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
