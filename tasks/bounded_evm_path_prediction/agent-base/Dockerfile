ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot, so Go is the snapshot's (1.22.2). The agent runs as uid 1000, the base image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git golang-go python3 \
    && install --directory --owner=1000 --group=1000 /workspace

# The Go module cache for the starter's modules, which the agent's sandbox has no network to
# fetch: `go mod download` runs once, here, as the source's recipe runs it. Go checks every
# module it fetches against `go.sum` and, since `GOFLAGS` leaves its default `-mod=readonly`,
# fails on a module that `go.sum` lacks instead of adding it, so the cache holds exactly the
# modules `go.sum` pins by hash. The build and the vet that follow prove the cache complete; their
# output is not kept.
RUN --mount=type=bind,source=repo,target=/frontier-repo,readonly \
    cp --archive /frontier-repo /tmp/modules \
    && cd /tmp/modules \
    && export GOMODCACHE=/opt/frontier/go-mod GOFLAGS=-modcacherw GOSUMDB=off GOTOOLCHAIN=local GOCACHE=/tmp/go-build \
    && go mod download \
    && GOPROXY=off go build ./... \
    && GOPROXY=off go vet ./core/vm \
    && rm --recursive --force /tmp/modules /tmp/go-build \
    && chmod --recursive a-w,a+rX /opt/frontier/go-mod

LABEL org.opencontainers.image.title="FrontierCS2 task bounded_evm_path_prediction" \
      dev.frontier.problem-id="bounded_evm_path_prediction"

# The solver-visible tree, and nothing else: the evaluator, the reference, the calibration, the
# snapshot and the vendored modules are in the verifier's image only. The repository's
# submission tools replace the base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GOMODCACHE=/opt/frontier/go-mod GOPROXY=file:///opt/frontier/go-mod/cache/download GOSUMDB=off GOTOOLCHAIN=local
WORKDIR /workspace
