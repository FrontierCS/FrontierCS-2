# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:22.23.3-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*
# OpenCode's grep/glob tools use ripgrep 15.1.0 and download this exact archive
# when no rg is on PATH; ship it so the Agent never fetches it.
ADD --checksum=sha256:1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599 \
    https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz /tmp/ripgrep.tar.gz
WORKDIR /runtime
COPY package.json package-lock.json ./
# opencode-ai is the npm wrapper (license, version); opencode-linux-x64 is the
# Bun-compiled glibc CLI it would select on this platform.
RUN npm ci --omit=dev --omit=optional --ignore-scripts --no-audit --no-fund \
    && grep -Fq '"version": "1.19.5"' node_modules/opencode-ai/package.json
RUN set -eu; \
    runtime=/opt/frontier/candidate/opencode; \
    install -d -m 0555 "$runtime/lib" "$runtime/bin"; \
    binary=/runtime/node_modules/opencode-linux-x64/bin/opencode; \
    cp "$binary" "$runtime/opencode"; \
    cp /runtime/node_modules/opencode-ai/LICENSE "$runtime/OPENCODE-LICENSE"; \
    cp /runtime/package.json /runtime/package-lock.json "$runtime/"; \
    cp -L /lib64/ld-linux-x86-64.so.2 "$runtime/lib/"; \
    ldd "$binary" \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" "$runtime/lib/"; done; \
    patchelf --set-interpreter /opt/frontier/candidate/opencode/lib/ld-linux-x86-64.so.2 \
        --set-rpath '$ORIGIN/lib' "$runtime/opencode"; \
    HOME=/tmp/probe XDG_DATA_HOME=/tmp/probe XDG_CACHE_HOME=/tmp/probe XDG_CONFIG_HOME=/tmp/probe \
        XDG_STATE_HOME=/tmp/probe OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 \
        "$runtime/opencode" --version | grep -Fx '1.19.5'; \
    rm -rf /tmp/probe; \
    mkdir /tmp/rg; tar -xzf /tmp/ripgrep.tar.gz -C /tmp/rg; \
    cp /tmp/rg/ripgrep-15.1.0-x86_64-unknown-linux-musl/rg "$runtime/bin/rg"; \
    cp /tmp/rg/ripgrep-15.1.0-x86_64-unknown-linux-musl/LICENSE-MIT "$runtime/RIPGREP-LICENSE-MIT"; \
    "$runtime/bin/rg" --version | head -1 | grep -F 'ripgrep 15.1.0'; \
    chmod 0555 "$runtime/opencode" "$runtime/bin/rg"; \
    install -d /out/opt/frontier/candidate; cp -a "$runtime" /out/opt/frontier/candidate/; \
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
