ARG VERIFIER_BASE
FROM ${VERIFIER_BASE} AS tools

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The packages that the pinned GCC 7.5 toolchain needs besides its own archive, as in the
# agent's image, and zstd for the evidence archive.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh build-essential git libgmp10 libmpc3 libmpfr6 python3 tar util-linux zlib1g zstd

# The pinned GCC 7.5/Cilk Plus toolchain, unpacked in a stage of its own (110 MB) and copied into
# `common` as one layer that both judges' images share. The archive is checked against the
# SHA-256 that `repo/frontier-toolchain.json` and `repo/frontier-cxx` pin, and is mounted, not
# copied, so that it is in no layer of this stage. The judge's isolate instances read the
# unpacked tree; they cannot read the task bundle. Every member is extracted as root's, as
# the source's recipe does: a rootless BuildKit cannot map the archive's owners (uid 3689121, gid
# 2626), and a failed chown makes tarfile skip the member's mode, so the compilers would lose their
# execute bits.
FROM tools AS gcc7
RUN --mount=type=bind,source=repo/frontier-gcc7.tar.xz,target=/context/frontier-gcc7.tar.xz \
    echo '13f10181b8e4a26f9e452a02ddbbf77620929b69f9d3eac09f3b3a27adae73b3  /context/frontier-gcc7.tar.xz' \
        | sha256sum --check --strict - \
    && python3 -c "import tarfile; tarfile.open('/context/frontier-gcc7.tar.xz').extractall('/usr/local/lib/frontier-gcc7', filter=lambda member, path: member.replace(uid=0, gid=0, uname='root', gname='root', deep=False))" \
    && chmod --recursive a+rX,a-w /usr/local/lib/frontier-gcc7

FROM tools AS common
COPY --from=gcc7 /usr/local/lib/frontier-gcc7 /usr/local/lib/frontier-gcc7

LABEL org.opencontainers.image.title="FrontierCS2 verifier parallel_radix_sort" \
      dev.frontier.problem-id="parallel_radix_sort"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The repository's verifier modules replace the base's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the reference and the
# std::sort baseline build from; the generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN --mount=type=bind,from=task-tools,source=/verifier-permissions.sh,target=/verifier-permissions.sh \
    sh /verifier-permissions.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The final workloads' hidden rows are for the final evaluation alone. The train judge's
# `evaluator.py` (`train/`) has both tables of them empty, and is laid over the judge in a stage
# of its own, from which the train judge's image copies the judge, so that the rows are in none
# of the train image's layers.
FROM common AS judge-train
COPY judge /judge
COPY train/ /judge/
RUN grep --quiet '^HIDDEN=\[\]$' /judge/evaluator.py \
    && grep --quiet '^LARGE_HIDDEN=\[\]$' /judge/evaluator.py \
    && ! grep --recursive --quiet '"f-' /judge

# The train judge's image: the judge with the hidden rows left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
