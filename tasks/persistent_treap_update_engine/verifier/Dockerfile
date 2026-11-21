FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The final suite's seeds come from a key that the final judge's `evaluator.py` reads
# (`_seed`). The train judge's `evaluator.py` (`train/`) has no such branch, and is laid over
# the judge in a stage of its own, from which the train judge's image copies the judge, so
# that the key's name, its default and the final suite's label are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN echo "ef12f0addbd91ab71dd33517346207b537483df9d12b4970d39c051e8d76baae  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge without the final suite's seed material.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
