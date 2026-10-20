# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:22.23.3-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS builder
# Step Code publishes no npm package, tag or release. The overlay builds the
# official repository from one exact commit with its own pnpm lockfile.
ARG STEPCODE_REPOSITORY=https://github.com/stepfun-ai/Step-Code.git
ARG STEPCODE_COMMIT=d0ec996df81f4089054f2cc80f1ae11139761f81
ARG STEPCODE_VERSION=0.1.0
# The repository's packageManager field.
ARG PNPM_VERSION=9.15.9
RUN apt-get update && apt-get install -y --no-install-recommends patchelf git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN npm install --global --no-audit --no-fund "pnpm@${PNPM_VERSION}" \
    && pnpm --version | grep -Fx "${PNPM_VERSION}"
WORKDIR /stepcode
RUN set -eu; \
    git init -q .; \
    git remote add origin "${STEPCODE_REPOSITORY}"; \
    git fetch -q --depth 1 origin "${STEPCODE_COMMIT}"; \
    git checkout -q --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${STEPCODE_COMMIT}"; \
    rm -rf .git
# Lifecycle scripts stay disabled: the repository's only root script installs
# git hooks, and the CLI needs no native postinstall step. build:offline uses
# the model metadata committed at this commit instead of downloading a catalog.
# After building, the dependency tree is relinked from the same lockfile and
# store with production dependencies only.
RUN pnpm install --frozen-lockfile --ignore-scripts \
    && pnpm run build:offline \
    && find . -name node_modules -type d -prune -exec rm -rf {} + \
    && CI=true pnpm install --frozen-lockfile --ignore-scripts --prod --offline \
    && HOME=/tmp/stepcode-version-home STEPCODE_DISABLE_PI_SERVICES=1 \
       node apps/cli/dist/main.js --version | grep -Fx "${STEPCODE_VERSION}" \
    && rm -rf /tmp/stepcode-version-home
RUN set -eu; \
    install -d -m 0555 /out/opt/frontier/candidate/stepcode/app \
        /out/opt/frontier/candidate/stepcode/lib; \
    cp /usr/local/bin/node /out/opt/frontier/candidate/stepcode/node; \
    cp /usr/local/LICENSE /out/opt/frontier/candidate/stepcode/NODE-LICENSE; \
    printf '%s\n' "${STEPCODE_REPOSITORY}" "${STEPCODE_COMMIT}" \
        > /out/opt/frontier/candidate/stepcode/SOURCE; \
    cp -a /stepcode/. /out/opt/frontier/candidate/stepcode/app/; \
    cp -L /lib64/ld-linux-x86-64.so.2 /out/opt/frontier/candidate/stepcode/lib/; \
    ldd /usr/local/bin/node \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" /out/opt/frontier/candidate/stepcode/lib/; done; \
    patchelf --set-interpreter /opt/frontier/candidate/stepcode/lib/ld-linux-x86-64.so.2 \
        --set-rpath '$ORIGIN/lib' /out/opt/frontier/candidate/stepcode/node; \
    chmod -R a+rX,go-w /out
COPY --chmod=0555 launcher.py /out/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 submit/SKILL.md /out/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
RUN find /out -type d -exec chmod 0555 {} +
FROM scratch
COPY --from=builder /out/ /
