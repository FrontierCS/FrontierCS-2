# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim@sha256:25330af3531fb5e23318554a0aa911125b6e91b1b777edf7655501d207c067a2 AS builder

ADD --checksum=sha256:dfb3096c91f4767c4d98006460800b7ba906a0b1a408280a926a8dc19a1af64f \
    https://lookaside.facebook.com/lookaside/muse/download/?channel=muse&version=1.4.2-R4684.1&file=muse-x86-linux /muse
RUN test "$(wc -c < /muse)" = 344369368 \
    && chmod 0555 /muse \
    && /muse --version | grep -F 'Muse Code 1.4.2 (1.4.2-R4684.1)'

RUN set -eu; \
    install -d -m 0555 /out/opt/frontier/candidate/muse-code/lib \
      /out/opt/frontier/candidate/bin \
      /out/opt/frontier/candidate/skills/frontier-submit \
      /out/opt/frontier/candidate/skills/frontier-gpu; \
    cp /muse /out/opt/frontier/candidate/muse-code/muse; \
    cp /usr/local/bin/node /out/opt/frontier/candidate/muse-code/node; \
    cp -L /lib64/ld-linux-x86-64.so.2 /out/opt/frontier/candidate/muse-code/lib/; \
    ldd /usr/local/bin/node \
      | awk '/=> \/|^\// {for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | while read -r library; do cp -L "$library" /out/opt/frontier/candidate/muse-code/lib/; done

COPY --chmod=0555 launch /out/opt/frontier/candidate/launch
COPY --chmod=0444 launcher.mjs transport.mjs /out/opt/frontier/candidate/muse-code/
COPY --chmod=0444 model-catalog.json /out/opt/frontier/candidate/muse-code/
COPY --chmod=0555 submit/frontier_submit.py /out/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 submit/SKILL.md /out/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/opt/frontier/candidate/skills/frontier-gpu/SKILL.md

# Catalog rows carry no invented host metadata; the launcher serves only the Candidate alias's row.
FROM scratch
COPY --from=builder /out/ /
