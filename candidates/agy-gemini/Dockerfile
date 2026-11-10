# syntax=docker/dockerfile:1.7
# Official Google Antigravity CLI (agy), pinned to the exact release the
# official installer (https://antigravity.google/cli/install.sh) resolved on
# 2026-10-24 from its linux_amd64 manifest, verified by the manifest's SHA-512.
ARG FRONTIER_AGY_BUILD_IMAGE=debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
FROM --platform=linux/amd64 ${FRONTIER_AGY_BUILD_IMAGE} AS agy-builder

ARG AGY_VERSION=1.4.7
ARG AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.4.7-5900205422411776/linux-x64/cli_linux_x64.tar.gz
ARG AGY_SHA512=beda760587589f5827b0b516c3ec2c5bd5cd2e91eb0e2533807ba1e990a27dcd63aef0191e0b76fb60feab0d16eaa8210fb79842c34dd4809e257e1712edd152

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    install -d /download /out/overlay/opt/frontier/candidate/agy/lib; \
    curl -fsSL --retry 3 -o /download/agy.tar.gz "$AGY_URL"; \
    echo "$AGY_SHA512  /download/agy.tar.gz" | sha512sum -c -; \
    tar -xzf /download/agy.tar.gz -C /download antigravity; \
    cp /download/antigravity /out/overlay/opt/frontier/candidate/agy/antigravity; \
    cp -L /lib64/ld-linux-x86-64.so.2 /out/overlay/opt/frontier/candidate/agy/lib/; \
    ldd /download/antigravity \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" /out/overlay/opt/frontier/candidate/agy/lib/; done; \
    patchelf --set-interpreter /opt/frontier/candidate/agy/lib/ld-linux-x86-64.so.2 \
        --set-rpath '$ORIGIN/lib' /out/overlay/opt/frontier/candidate/agy/antigravity; \
    install -d /opt/frontier; ln -s /out/overlay/opt/frontier/candidate /opt/frontier/candidate; \
    HOME=/tmp/agy-version /opt/frontier/candidate/agy/antigravity --help 2>&1 | grep -q -- '--dangerously-skip-permissions'; \
    rm /opt/frontier/candidate; \
    printf '%s\n' "version=$AGY_VERSION" "url=$AGY_URL" "sha512=$AGY_SHA512" \
        > /out/overlay/opt/frontier/candidate/agy/AGY-RELEASE; \
    chmod -R a+rX,go-w /out/overlay

COPY --chmod=0555 launcher.py /out/overlay/opt/frontier/candidate/launch
COPY --chmod=0444 gemini_chat_bridge.py /out/overlay/opt/frontier/candidate/agy/gemini_chat_bridge.py
COPY --chmod=0555 submit/frontier_submit.py /out/overlay/opt/frontier/candidate/agy/tools/frontier_submit.py
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/overlay/opt/frontier/candidate/agy/tools/frontier_gpu.py
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/overlay/opt/frontier/candidate/agy/tools/frontier_gpu_connect_proxy.py
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 platform_tool.py /out/overlay/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
RUN install -d -m 0555 /out/overlay/opt/frontier/candidate/skills \
    /out/overlay/opt/frontier/candidate/skills/frontier-submit \
    /out/overlay/opt/frontier/candidate/skills/frontier-gpu
COPY --chmod=0444 submit/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/overlay/opt/frontier/candidate/skills/frontier-gpu/SKILL.md

FROM scratch
COPY --from=agy-builder /out/overlay/ /
