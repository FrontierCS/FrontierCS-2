# The verifier base, which a task's verifier image starts from as `VERIFIER_BASE`: Ubuntu 24.04
# by digest, apt from one dated snapshot, Python with zstandard, the uid 65532 user, the shared
# verifier (/opt/frontier/{verify,fcs2_verifier,beat_spec_feedback}.py), the runtime-python and
# /opt/frontier/verifier launchers, the shared patch applier, and the isolate launcher
# (/opt/frontier/isolate.py: a judge runs submitted code through isolate.run).
# `install_frontier_pth.py` puts /opt/frontier on python3's default import path (frontier.pth),
# so every judge process, workers included, imports these modules by name. It is FrontierCS's
# `frontier-verifier-base` recipe, built here with the repository's `shared/` in its context. A
# task's own recipe adds its packages, judge/ and the generated config.yaml and verifier.json,
# then sets the final permissions, USER, ENV and WORKDIR.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS certificates

FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

USER root
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
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir zstandard==0.25.0

RUN install -d -o 65532 -g 65532 /frontier \
    && if ! getent passwd 65532 >/dev/null; then useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier; fi
ENV HOME=/frontier

COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py shared/verifier/beat_spec_feedback.py \
     shared/lib/edit_scope.py shared/lib/isolate.py /opt/frontier/
RUN --mount=type=bind,source=shared/lib/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
    python3 /tmp/install_frontier_pth.py /opt/frontier
RUN python3 -c 'import pathlib, shlex, sys; path = pathlib.Path("/opt/frontier/runtime-python"); output = path.open("x"); output.write("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " \"$@\"\n"); output.close(); path.chmod(0o555)' \
    && /opt/frontier/runtime-python -c 'import zstandard' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0555 /opt/frontier/verifier \
    && find /opt/frontier -maxdepth 2 -type f -name '*.py' -exec chmod 0444 {} +

LABEL org.opencontainers.image.title="FrontierCS2 shared Verifier base" \
      dev.frontier.apt-snapshot="${APT_SNAPSHOT}"
