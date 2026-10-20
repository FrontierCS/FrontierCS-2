# syntax=docker/dockerfile:1.7

ARG FRONTIER_NODE_BUILD_IMAGE=node:22-bookworm-slim

FROM ${FRONTIER_NODE_BUILD_IMAGE} AS codex-builder
ARG TARGETARCH

# Official native package, including its rg/bwrap/runtime closure. The package
# digest binds the entire closure rather than only the executable version.
ADD --checksum=sha256:7417711678b464eee70664126e39e88bb3c84c2928da0e71f14b56e04ff5c09f \
    https://registry.npmjs.org/@openai/codex/-/codex-0.162.1-linux-x64.tgz /tmp/codex.tgz
RUN set -eu; \
    test "${TARGETARCH}" = amd64; \
    mkdir -p /tmp/codex-package /out; \
    tar -xzf /tmp/codex.tgz -C /tmp/codex-package; \
    cp -R /tmp/codex-package/package/vendor/x86_64-unknown-linux-musl /out/codex; \
    chmod 0555 /out/codex/bin/codex; \
    echo '628e35487d888a28f0427fc37501da529dd3d81fcbad7a2c3421c1e8670d92e1  /out/codex/bin/codex' | sha256sum -c -; \
    /out/codex/bin/codex --version | grep -Fx 'codex-cli 0.162.1'

# BuildKit applies --chmod to implicitly-created parent directories as well as
# the copied file.  Build the non-Codex overlay tree explicitly so every
# directory remains searchable by the unprivileged Candidate user while the
# files keep their read/execute-only modes.
RUN install -d -m 0555 \
        /out/overlay/opt/frontier/candidate/bin \
        /out/overlay/opt/frontier/candidate/skills/frontier-gpu \
        /out/overlay/opt/frontier/candidate/skills/frontier-submit \
    && test -z "$(find /out/overlay/opt/frontier/candidate -type d ! -perm /0111 -print -quit)"
COPY --chmod=0555 launcher.py /out/overlay/opt/frontier/candidate/launch
COPY --chmod=0555 frontier_submit.py /out/overlay/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 frontier_gpu.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 frontier_gpu_connect_proxy.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 skills/frontier-gpu/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
COPY --chmod=0444 skills/frontier-submit/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-submit/SKILL.md

# This is a filesystem overlay. The Task Agent base supplies Python, CA roots,
# user, workdir and the rest of the task environment.
FROM scratch

COPY --from=codex-builder /out/codex /opt/frontier/candidate/codex
COPY --from=codex-builder /out/overlay/ /
