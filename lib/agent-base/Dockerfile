# The agent base, which every task's agent image starts from as `AGENT_BASE`: Ubuntu 24.04 by
# digest, apt from one dated snapshot, the uid 10001 user and /workspace, and the shared agent
# tools under /frontier/task/bin. It is FrontierCS's `frontier-agent-base` recipe, built here
# with the repository's `shared/` in its context. A task's own recipe adds its packages,
# README.md, repo/ and generated submission-policy.json, then sets the final permissions, USER,
# ENV and WORKDIR.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS certificates

FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_SNAPSHOT=20261103T000000Z
# The base holds no CA certificates, which the snapshot's HTTPS needs before `ca-certificates`
# can be installed, so the bundle of a CUDA base of the same Ubuntu release, pinned by digest, is
# copied in first; the package replaces it.
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN printf 'Types: deb\nURIs: https://snapshot.ubuntu.com/ubuntu/%s\nSuites: noble noble-updates noble-backports noble-security\nComponents: main restricted universe multiverse\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n' "$APT_SNAPSHOT" \
       > /etc/apt/sources.list.d/ubuntu.sources \
    && printf 'Acquire::Retries "10";\nAcquire::Retries::Delay::Maximum "60";\n' > /etc/apt/apt.conf.d/80-frontier-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace

# The submission tools run on this base's /usr/bin/python3 and git, never on a Python the task
# adds; edit_scope.py is the same file the verifier base carries.
COPY shared/agent/bin /frontier/task/bin
COPY shared/lib/edit_scope.py /frontier/task/bin/edit_scope.py
RUN ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh

LABEL org.opencontainers.image.title="FrontierCS2 shared Agent base" \
      dev.frontier.apt-snapshot="${APT_SNAPSHOT}"
