FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24 AS packages
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_XMLDOC_MODE=skip NUGET_PACKAGES=/opt/frontier/nuget
COPY agent-base/frontier/task/repo/project /build
RUN dotnet restore /build/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj && chmod -R a+rX,a-w /opt/frontier/nuget
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24
ENV DEBIAN_FRONTEND=noninteractive DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=/opt/frontier/nuget
RUN apt-get update && apt-get install -y --no-install-recommends python3 git openssh-client tar && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier && install -d -o frontier -g frontier /workspace
COPY --from=packages /opt/frontier/nuget /opt/frontier/nuget
COPY agent-base/frontier/task /frontier/task
RUN chmod -R a+rX,a-w /frontier/task /opt/frontier/nuget
USER frontier
WORKDIR /workspace

# The NuGet vulnerability audit queries api.nuget.org, and the repository treats
# its warning as an error. The Agent sandbox has no network access; packages
# restore from the baked cache above.
ENV NuGetAudit=false
