ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential git tar zstd
# The source's images compiled with GCC 12 (Debian 12.2.0); the shared Ubuntu base ships GCC 13,
# which
# measurably changed one historical submission's timings. GCC 12 is installed and linked as
# `gcc`/`g++`/`cc`/`c++`/`cpp` for the agent, the Makefile and the judge, fetched as above.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh gcc-12 g++-12 cpp-12 \
    && for tool in gcc g++ cpp gcov gcc-ar gcc-nm gcc-ranlib; do ln --symbolic --force "$tool-12" "/usr/bin/$tool"; done \
    && ln --symbolic --force gcc-12 /usr/bin/cc && ln --symbolic --force g++-12 /usr/bin/c++ \
    && test "$(g++ -dumpversion | cut -d. -f1)" = 12 && test "$(cc -dumpversion | cut -d. -f1)" = 12 \
    && test "$(c++ -dumpversion | cut -d. -f1)" = 12 && test "$(cpp -dumpversion | cut -d. -f1)" = 12

LABEL org.opencontainers.image.title="FrontierCS2 verifier parallel_dbscan" \
      dev.frontier.problem-id="parallel_dbscan"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final suite's seeds are for the final evaluation alone. The train judge's `evaluator.py`
# (`train/`) has none, and is laid over the judge in a stage of its own, from which the train
# judge's image copies the judge, so that the seeds are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN echo "68e9e6276537e140253c196215e68949276f5a1df6e31dc7c23bbc1fd03c7196  /judge/evaluator.py" | sha256sum --check --strict -

# The train judge's image: the judge with the final suite's seeds left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
