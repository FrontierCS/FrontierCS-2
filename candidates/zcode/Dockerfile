# syntax=docker/dockerfile:1.7
ARG FRONTIER_NODE_BUILD_IMAGE=node:24.14.0-bookworm-slim@sha256:d8e448a56fc63242f70026718378bd4b00f8c82e78d20eefb199224a4d8e33d8
FROM ${FRONTIER_NODE_BUILD_IMAGE} AS builder
ARG ZCODE_COMMIT=29628c9acdb81b703bbd4080c207a0e7ce5e276e
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global pnpm@10.33.2
WORKDIR /source
RUN git init && git remote add origin https://github.com/zai-org/ZCode.git \
    && git fetch --depth=1 origin "${ZCODE_COMMIT}" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${ZCODE_COMMIT}" \
    && pnpm install --frozen-lockfile --ignore-scripts --filter @zcode/cli... \
    && pnpm exec turbo --skip-infer --cwd apps/zcode-cli run build --filter @zcode/cli --concurrency 3 \
    && node apps/zcode-cli/packages/cli/dist/zcode.cjs --version
RUN install -d /out/opt/frontier/candidate/zcode/app /out/opt/frontier/candidate/zcode/lib \
        /out/opt/frontier/candidate/bin /out/opt/frontier/candidate/skills/frontier-submit \
        /out/opt/frontier/candidate/skills/frontier-gpu \
    && cp /usr/local/bin/node /out/opt/frontier/candidate/zcode/node \
    && cp -L /lib64/ld-linux-x86-64.so.2 /out/opt/frontier/candidate/zcode/lib/ \
    && ldd /usr/local/bin/node | awk '/=> \/|^\// {for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' \
       | sort -u | while read -r library; do cp -L "$library" /out/opt/frontier/candidate/zcode/lib/; done \
    && patchelf --set-interpreter /opt/frontier/candidate/zcode/lib/ld-linux-x86-64.so.2 \
         --set-rpath '$ORIGIN/lib' /out/opt/frontier/candidate/zcode/node \
    && cp apps/zcode-cli/packages/cli/dist/zcode.cjs /out/opt/frontier/candidate/zcode/app/ \
    && cp apps/zcode-cli/packages/cli/dist/THIRD-PARTY-NOTICES.md /out/opt/frontier/candidate/zcode/app/ \
    && cp -R apps/zcode-cli/packages/cli/dist/provider /out/opt/frontier/candidate/zcode/app/ \
    && cp LICENSE /out/opt/frontier/candidate/zcode/app/LICENSE \
    && cp /usr/local/LICENSE /out/opt/frontier/candidate/zcode/app/NODE-LICENSE \
    && printf '%s\n' "$ZCODE_COMMIT" > /out/opt/frontier/candidate/zcode/app/SOURCE_COMMIT
COPY --chmod=0555 launcher.py /out/opt/frontier/candidate/launch
COPY --chmod=0444 fetch-sanitize.cjs /out/opt/frontier/candidate/zcode/fetch-sanitize.cjs
COPY --chmod=0555 submit/frontier_submit.py /out/opt/frontier/candidate/bin/frontier-submit
COPY --chmod=0555 gpu-helper/frontier_gpu.py /out/opt/frontier/candidate/bin/frontier-gpu
COPY --chmod=0555 gpu-helper/frontier_gpu_connect_proxy.py /out/opt/frontier/candidate/bin/frontier-gpu-connect-proxy
COPY --chmod=0444 submit/SKILL.md /out/opt/frontier/candidate/skills/frontier-submit/SKILL.md
COPY --chmod=0444 gpu-helper/SKILL.md /out/opt/frontier/candidate/skills/frontier-gpu/SKILL.md
FROM scratch
COPY --from=builder /out/ /
