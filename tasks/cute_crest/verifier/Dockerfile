ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The judge's packages, fetched as in the agent's image.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh binutils bsdutils build-essential ca-certificates coreutils dpkg git libncurses-dev make patch perl python3 python3-pip time

LABEL org.opencontainers.image.title="FrontierCS2 verifier cute_crest" \
      dev.frontier.problem-id="cute_crest"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer. Each judge's
# image is also its search machine's, whose commands run in /workspace, as FrontierCS's CPU runner
# layer makes it, owned by the machine's user.
RUN ln --symbolic /pumpkin/out /frontier/output \
    && install --directory --owner=65532 --group=65532 /workspace

# The shared verifier: the repository's `shared/` ones, which replace the base's; and the port's
# adapter through which the judge leases its search machine.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /verifier-lease/fcs2_verifier_gpu.py /opt/frontier/
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

# The harness and the instrumented programs, built once from the whole judge as the source
# builds them, outside the judge's directory (a verification copies that, not this). The build
# fetches the program sources by pinned URL and SHA-256 (`judge/benchmark_sources.json`), and
# they stay in the image beside the binaries built from them, with their licence files
# (`/opt/crest-task/src`). Vim and sed are built in every build, the dev program's with them.
FROM common AS built
COPY --chmod=a+rX,a-w judge /opt/frontier/task
RUN python3 /opt/frontier/task/build_benchmarks.py /opt/crest-task

# The final programs (Vim 5.7, sed 3.02), their built binaries and sources, the recipe that
# builds them and the final list of programs are for the final evaluation alone. The train
# judge's copy of the judge is made without them, in a stage of its own, with a `programs.json`
# whose `final` is empty (`train/programs.json`), from which the train judge's image copies it,
# so that they are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/programs.json /judge/programs.json
RUN rm --recursive --force /judge/benchmarks /judge/benchmark_sources.json /judge/build_benchmarks.py
RUN test ! -e /judge/benchmarks && test ! -e /judge/benchmark_sources.json && test ! -e /judge/build_benchmarks.py

# What both images hold of the build: the harness, its toolchain and the development program,
# in one stage, so that the two images share its layers.
FROM common AS crest
COPY --from=built /opt/crest-task/tree /opt/crest-task/tree
COPY --from=built /opt/crest-task/build /opt/crest-task/build
COPY --from=built /opt/crest-task/bench/dev /opt/crest-task/bench/dev
COPY --from=built /opt/crest-task/src/grep-2.2 /opt/crest-task/src/grep-2.2

# The train judge's image: the judge without the final programs, and the build above, not the
# final programs or their sources.
FROM crest AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is, with everything built.
FROM crest AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
COPY --from=built /opt/crest-task/bench/final_a /opt/crest-task/bench/final_a
COPY --from=built /opt/crest-task/bench/final_b /opt/crest-task/bench/final_b
COPY --from=built /opt/crest-task/src/sed-3.02 /opt/crest-task/src/sed-3.02
COPY --from=built /opt/crest-task/src/vim-5.7 /opt/crest-task/src/vim-5.7
USER 65532:65532
