# syntax=docker/dockerfile:1.7

ARG KIMI_CODE_VERSION=0.31.0
ARG FRONTIER_NODE_BUILD_IMAGE=node:22-bookworm-slim

FROM ${FRONTIER_NODE_BUILD_IMAGE} AS kimi-code-builder
ARG KIMI_CODE_VERSION

RUN npm install --global --omit=dev "@moonshot-ai/kimi-code@${KIMI_CODE_VERSION}" \
    && kimi --version | grep -F "${KIMI_CODE_VERSION}"

# Kimi 0.31.0 otherwise rejects foreground Bash timeouts above 300 seconds.
# Preserve the proven Frontier CLI tool runtime by widening this pinned
# constant, and fail closed if the
# upstream bundle shape changes.
RUN set -eu; \
    main=/usr/local/lib/node_modules/@moonshot-ai/kimi-code/dist/main.mjs; \
    test "$(grep -Fc 'MAX_TIMEOUT_S$1 = 300;' "${main}")" -eq 1; \
    sed -i 's/MAX_TIMEOUT_S$1 = 300;/MAX_TIMEOUT_S$1 = 3600;/' "${main}"; \
    test "$(grep -Fc 'MAX_TIMEOUT_S$1 = 3600;' "${main}")" -eq 1; \
    ! grep -Fq 'MAX_TIMEOUT_S$1 = 300;' "${main}"

RUN set -eu; \
    install -d -m 0555 \
        /out/overlay/opt/frontier/candidate/kimi-code/app \
        /out/overlay/opt/frontier/candidate/kimi-code/lib \
        /out/overlay/opt/frontier/candidate/bin \
        /out/overlay/opt/frontier/candidate/skills/frontier-gpu \
        /out/overlay/opt/frontier/candidate/skills/frontier-submit; \
    cp /usr/local/bin/node /out/overlay/opt/frontier/candidate/kimi-code/node; \
    cp -L /lib64/ld-linux-x86-64.so.2 \
        /out/overlay/opt/frontier/candidate/kimi-code/lib/ld-linux-x86-64.so.2; \
    cp -R /usr/local/lib/node_modules/@moonshot-ai/kimi-code/. \
        /out/overlay/opt/frontier/candidate/kimi-code/app/; \
    ldd /usr/local/bin/node \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "${library}" /out/overlay/opt/frontier/candidate/kimi-code/lib/; done; \
    chmod 0555 /out/overlay/opt/frontier/candidate/kimi-code/node; \
    test -f /out/overlay/opt/frontier/candidate/kimi-code/app/dist/main.mjs

COPY --chmod=0555 launcher.py /out/overlay/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/overlay/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 gpu-helper/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
COPY --chmod=0444 submit/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-submit/SKILL.md

FROM scratch

COPY --from=kimi-code-builder /out/overlay/ /
