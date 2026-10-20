FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24 AS packages
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_XMLDOC_MODE=skip NUGET_PACKAGES=/opt/frontier/nuget
COPY agent-base/frontier/task/repo/project /build
RUN dotnet restore /build/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj && chmod -R a+rX,a-w /opt/frontier/nuget
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24
ENV DEBIAN_FRONTEND=noninteractive DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=/opt/frontier/nuget PYTHONDONTWRITEBYTECODE=1 TMPDIR=/frontier/tmp
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-numpy python3-scipy python3-zstandard tar && rm -rf /var/lib/apt/lists/* && python3 -m pip install --break-system-packages --no-cache-dir gurobipy==13.0.0 && install -d -o 65532 -g 65532 /frontier
COPY --from=packages /opt/frontier/nuget /opt/frontier/nuget
COPY judge /opt/frontier/task
COPY verifier/verify.py /opt/frontier/verify.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py
COPY verifier/patch_payload.py /opt/frontier/patch_payload.py
COPY agent-base/frontier/task/repo/project /opt/frontier/starter/project
RUN printf '#!/bin/sh\nexec python3 /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier && chmod 0555 /opt/frontier/verifier && chmod 0444 /opt/frontier/verify.py /opt/frontier/fcs2_verifier.py /opt/frontier/beat_spec_feedback.py /opt/frontier/patch_payload.py && chmod -R a+rX,a-w /opt/frontier/task /opt/frontier/nuget /opt/frontier/starter
# .NET named mutexes use /tmp/.dotnet even when TMPDIR is set.
RUN rm -rf /tmp/.dotnet && ln -s /frontier/tmp/.dotnet /tmp/.dotnet
USER 65532:65532
WORKDIR /frontier
