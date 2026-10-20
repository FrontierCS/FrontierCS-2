# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:22.23.3-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /runtime
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund \
    && node node_modules/@qwen-code/qwen-code/cli-entry.js --version | grep -Fx '0.25.0' \
    && chmod 0555 node_modules/@qwen-code/qwen-code/vendor/ripgrep/x64-linux/rg
RUN set -eu; \
    install -d -m 0555 /out/opt/frontier/candidate/qwen-code/app \
        /out/opt/frontier/candidate/qwen-code/lib; \
    cp /usr/local/bin/node /out/opt/frontier/candidate/qwen-code/node; \
    cp /usr/local/LICENSE /out/opt/frontier/candidate/qwen-code/NODE-LICENSE; \
    cp -R /runtime/. /out/opt/frontier/candidate/qwen-code/app/; \
    cp -L /lib64/ld-linux-x86-64.so.2 /out/opt/frontier/candidate/qwen-code/lib/; \
    ldd /usr/local/bin/node \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" /out/opt/frontier/candidate/qwen-code/lib/; done; \
    patchelf --set-interpreter /opt/frontier/candidate/qwen-code/lib/ld-linux-x86-64.so.2 \
        --set-rpath '$ORIGIN/lib' /out/opt/frontier/candidate/qwen-code/node; \
    chmod -R a+rX /out
COPY --chmod=0555 launcher.py /out/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 submit/SKILL.md /out/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
RUN find /out -type d -exec chmod 0555 {} +
FROM scratch
COPY --from=builder /out/ /
