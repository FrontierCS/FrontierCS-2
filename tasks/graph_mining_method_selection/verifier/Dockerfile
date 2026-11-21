FROM verifier-base AS common
ARG DEBIAN_FRONTEND=noninteractive

# The measured table's rows 112 to 223 are the final suite's answers, for the final evaluation
# alone. The train judge's table (`train/`) holds rows 0 to 111 twice, so that the evaluator's
# check of the table's length passes and no row of the final suite is in it, and is laid over
# the judge in a stage of its own, from which the train judge's image copies the judge, so
# that the final rows are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN python3 -I -c "import json; rows = json.load(open('/judge/measured_workloads.json')); assert len(rows) == 224 and rows[112:] == rows[:112]"

# The train judge's image: the judge with the table of its first 112 rows twice.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
