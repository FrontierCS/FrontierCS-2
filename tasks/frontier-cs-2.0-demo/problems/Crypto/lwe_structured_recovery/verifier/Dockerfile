# syntax=docker/dockerfile:1.7
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

LABEL org.opencontainers.image.title="FrontierCS2 structured-LWE verifier" \
      org.opencontainers.image.licenses="MIT" \
      dev.frontier.problem-id="lwe_structured_recovery" \
      dev.frontier.upstream-commit="15ce5cc28f69a79bc3555bc4c8de65424360b77f" \
      dev.frontier.catalog-sha256="318f32b68e8a30c6bcd5867c89eb10d87a5472614401f4e6df0f55356cdc98a6"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-zstandard \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-create-home --uid 65532 --shell /usr/sbin/nologin frontier-verifier \
    && install -d -o 65532 -g 65532 /frontier

COPY agent-base/frontier/task/repo /opt/frontier/lwe/repo
COPY verifier/verify.py /opt/frontier/lwe-verifier
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/patch_payload.py /opt/frontier/patch_payload.py
COPY agent-base/frontier/task/starter /opt/frontier/lwe/starter
COPY judge /opt/frontier/lwe/judge
COPY LICENSE.frontier-cs /opt/frontier/LICENSE.frontier-cs

RUN chmod 0555 /opt/frontier/lwe-verifier \
    && chmod 0444 /opt/frontier/patch_payload.py \
    && find /opt/frontier/lwe -type d -exec chmod 0555 {} + \
    && find /opt/frontier/lwe -type f -exec chmod 0444 {} +

USER 65532:65532
ENV HOME=/frontier TMPDIR=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTIER_LWE_ROOT=/opt/frontier/lwe FRONTIER_LWE_JUDGE=/opt/frontier/lwe/judge
ENTRYPOINT ["/opt/frontier/lwe-verifier"]
