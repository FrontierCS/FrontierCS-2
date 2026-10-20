# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:22.23.3-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS deepseek-builder

RUN apt-get update && apt-get install -y --no-install-recommends patchelf

WORKDIR /runtime
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund \
    && node node_modules/@deepseek-ai/dsh/lib/bin.js --version | grep -Fx '0.2.0-rc.2'
COPY patch_transport_error.mjs /runtime/patch_transport_error.mjs
RUN node /runtime/patch_transport_error.mjs node_modules/@deepseek-ai/dsh-llm-pi-ai/lib/index.js
RUN set -eu; \
    install -d -m 0555 /out/overlay/opt/frontier/candidate/deepseek-harness/app \
        /out/overlay/opt/frontier/candidate/deepseek-harness/lib; \
    cp /usr/local/bin/node /out/overlay/opt/frontier/candidate/deepseek-harness/node; \
    cp /usr/local/LICENSE /out/overlay/opt/frontier/candidate/deepseek-harness/NODE-LICENSE; \
    cp -R /runtime/. /out/overlay/opt/frontier/candidate/deepseek-harness/app/; \
    cp -L /lib64/ld-linux-x86-64.so.2 /out/overlay/opt/frontier/candidate/deepseek-harness/lib/; \
    ldd /usr/local/bin/node \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" /out/overlay/opt/frontier/candidate/deepseek-harness/lib/; done; \
    patchelf --set-interpreter /opt/frontier/candidate/deepseek-harness/lib/ld-linux-x86-64.so.2 \
        --set-rpath '$ORIGIN/lib' /out/overlay/opt/frontier/candidate/deepseek-harness/node; \
    chmod -R a+rX /out/overlay

COPY --chmod=0555 launcher.py /out/overlay/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/overlay/opt/frontier/candidate/deepseek-harness/tools/frontier_submit.py
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/overlay/opt/frontier/candidate/deepseek-harness/tools/frontier_gpu.py
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/overlay/opt/frontier/candidate/deepseek-harness/tools/frontier_gpu_connect_proxy.py
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
RUN install -d -m 0555 /out/overlay/opt/frontier/candidate/skills \
    /out/overlay/opt/frontier/candidate/skills/frontier-submit \
    /out/overlay/opt/frontier/candidate/skills/frontier-gpu
COPY --chmod=0444 submit/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-gpu/SKILL.md

FROM scratch
COPY --from=deepseek-builder /out/overlay/ /
