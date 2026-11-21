ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages. Go 1.23.12 and the estimator's module cache come from the toolchain image
# (`toolchain/Dockerfile`). The agent runs as uid 1000, the base image's `ubuntu`, which owns
# /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates curl git python3 tar time \
    && install --directory --owner=1000 --group=1000 /workspace
COPY --from=go-toolchain /usr/local/go /usr/local/go
COPY --from=go-toolchain /opt/frontier/go-mod /opt/frontier/go-mod
RUN ln --symbolic /usr/local/go/bin/go /usr/local/bin/go \
    && ln --symbolic /usr/local/go/bin/gofmt /usr/local/bin/gofmt

LABEL org.opencontainers.image.title="FrontierCS2 task ckks_composable_noise" \
      dev.frontier.problem-id="ckks_composable_noise"

# The solver-visible tree, and nothing else: the evaluator, reference and beat-reference
# specification are in the verifier's image only. The repository's submission tools replace the
# base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CKKS_MODULE_CACHE=/opt/frontier/go-mod GOMODCACHE=/opt/frontier/go-mod GOPROXY=off
WORKDIR /workspace
