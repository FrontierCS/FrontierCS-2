# syntax=docker/dockerfile:1.7

ARG KIMI_CODE_VERSION=2.3.3
ARG FRONTIER_NODE_BUILD_IMAGE=node:22-bookworm-slim

FROM ${FRONTIER_NODE_BUILD_IMAGE} AS kimi-code-builder
ARG KIMI_CODE_VERSION

RUN npm install --global --omit=dev "@moonshot-ai/kimi-code@${KIMI_CODE_VERSION}" \
    && kimi --version | grep -F "${KIMI_CODE_VERSION}"

# Kimi 2.3.3 otherwise rejects foreground Bash timeouts above 300 seconds.
# Preserve the proven Frontier CLI tool runtime by widening the foreground cap
# to 3600 seconds. 2.x inlines the cap at three enforcement sites, one prompt
# variable and three model-facing description strings; patch each exact form
# and fail closed if the upstream bundle shape changes.
RUN set -eu; \
    main=/usr/local/lib/node_modules/@moonshot-ai/kimi-code/dist/main.mjs; \
    count() { grep -oF -- "$1" "${main}" | wc -l; }; \
    test "$(count 'isBackground ? MAX_BACKGROUND_TIMEOUT_S : 300;')" -eq 3; \
    test "$(count 'MAX_TIMEOUT_S: 300,')" -eq 1; \
    test "$(count '${String(300)}')" -eq 3; \
    sed -i \
        -e 's/isBackground ? MAX_BACKGROUND_TIMEOUT_S : 300;/isBackground ? MAX_BACKGROUND_TIMEOUT_S : 3600;/g' \
        -e 's/MAX_TIMEOUT_S: 300,/MAX_TIMEOUT_S: 3600,/' \
        -e 's/\${String(300)}/${String(3600)}/g' \
        "${main}"; \
    test "$(count 'isBackground ? MAX_BACKGROUND_TIMEOUT_S : 3600;')" -eq 3; \
    test "$(count 'MAX_TIMEOUT_S: 3600,')" -eq 1; \
    test "$(count '${String(3600)}')" -eq 3; \
    test "$(count 'MAX_BACKGROUND_TIMEOUT_S : 300;')" -eq 0; \
    test "$(count 'MAX_TIMEOUT_S: 300,')" -eq 0; \
    test "$(count '${String(300)}')" -eq 0

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
    test -f /out/overlay/opt/frontier/candidate/kimi-code/app/dist/main.mjs; \
    test -f /out/overlay/opt/frontier/candidate/kimi-code/app/dist/search-worker.mjs; \
    for dependency in ws qrcode chokidar; do \
        test -f "/out/overlay/opt/frontier/candidate/kimi-code/app/node_modules/${dependency}/package.json"; \
    done

COPY --chmod=0555 launcher.py /out/overlay/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/overlay/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 gpu-helper/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
COPY --chmod=0444 submit/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-submit/SKILL.md

FROM scratch

COPY --from=kimi-code-builder /out/overlay/ /
