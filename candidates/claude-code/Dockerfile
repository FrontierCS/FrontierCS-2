# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:24.14.0-bookworm-slim@sha256:d8e448a56fc63242f70026718378bd4b00f8c82e78d20eefb199224a4d8e33d8
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Fixed official native Linux/x64 package; never fetch "latest" at runtime.
RUN curl -fsSL https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.321.tgz -o /tmp/claude.tgz \
    && echo '956f8c8544e19d4da1afa42743dc600fc73a3960096d836988ccc843838968bc5c0724ece77bfbc15ff62331aca416dc3219f1168c75befa25b6e5b29bf87abb  /tmp/claude.tgz' | sha512sum -c - \
    && mkdir /source && tar -xzf /tmp/claude.tgz -C /source \
    && install -d /out/opt/frontier/candidate/claude-code/lib /out/opt/frontier/candidate/bin \
         /out/opt/frontier/candidate/skills/frontier-submit /out/opt/frontier/candidate/skills/frontier-gpu \
    && cp /source/package/claude /source/package/LICENSE.md /out/opt/frontier/candidate/claude-code/ \
    && cp -L /lib64/ld-linux-x86-64.so.2 /out/opt/frontier/candidate/claude-code/lib/ \
    && ldd /source/package/claude | awk '/=> \/|^\// {for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' \
       | sort -u | while read -r library; do cp -L "$library" /out/opt/frontier/candidate/claude-code/lib/; done \
    && /out/opt/frontier/candidate/claude-code/lib/ld-linux-x86-64.so.2 \
         --library-path /out/opt/frontier/candidate/claude-code/lib \
         /out/opt/frontier/candidate/claude-code/claude --version | grep -Fx '2.1.321 (Claude Code)'

COPY --chmod=0555 launcher.py /out/opt/frontier/candidate/launch
COPY --chmod=0555 submit/frontier_submit.py /out/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 submit/SKILL.md /out/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
FROM scratch
COPY --from=builder /out/ /
