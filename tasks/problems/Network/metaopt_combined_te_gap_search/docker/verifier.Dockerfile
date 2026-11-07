# Verifier image of metaopt_combined_te_gap_search, both modes: the shared Verifier base plus the
# reviewed .NET 8 SDK and offline NuGet cache of the old image, the reference LP stack (numpy and
# scipy from the base's Ubuntu, gurobipy 13.0.0 as before), this task's judge and the starter
# repo the shared verifier overlays the submission on. Build context: this task's root, with
# .frontier-generated/ added by the build.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The reviewed .NET SDK of the old images (mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim, SDK
# 8.0.425 with runtimes 8.0.31 and PowerShell 7.4.20), installed on the shared base from
# Microsoft's official artifacts with the same pins the MCR Dockerfile used (dotnet/dotnet-docker
# 5a37658d: SDK tarball sha512, PowerShell nupkg sha256), and the MCR runtime dependencies.
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_GENERATE_ASPNET_CERTIFICATE=false \
    DOTNET_SDK_VERSION=8.0.425 DOTNET_VERSION=8.0.31 ASPNET_VERSION=8.0.31 \
    DOTNET_USE_POLLING_FILE_WATCHER=true DOTNET_RUNNING_IN_CONTAINER=true \
    NUGET_XMLDOC_MODE=skip NUGET_PACKAGES=/opt/frontier/nuget NuGetAudit=false \
    POWERSHELL_DISTRIBUTION_CHANNEL=PSDocker-DotnetSDK-Debian-12 APP_UID=1654 ASPNETCORE_HTTP_PORTS=8080
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libatomic1 libc6 libgcc-s1 \
       libicu74 libssl3t64 libstdc++6 openssh-client python3 tar tzdata wget zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --show-error --location --output /tmp/dotnet.tar.gz \
       https://builds.dotnet.microsoft.com/dotnet/Sdk/8.0.425/dotnet-sdk-8.0.425-linux-x64.tar.gz \
    && echo "934b8060a7190e5909ad1fd0785db542f487b3bbf6cdd14826b02095fdd0d0394298b1634085eff302928fccc33f7c1a7253e9b87df555fc36fce819bcd2e798  /tmp/dotnet.tar.gz" | sha512sum -c - \
    && mkdir -p /usr/share/dotnet && tar -xzf /tmp/dotnet.tar.gz -C /usr/share/dotnet && rm /tmp/dotnet.tar.gz \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && dotnet help >/dev/null \
    && curl --fail --show-error --location --output /PowerShell.Linux.x64.7.4.20.nupkg \
       https://powershellinfraartifacts-gkhedzdeaghdezhr.z01.azurefd.net/tool/7.4.20/PowerShell.Linux.x64.7.4.20.nupkg \
    && echo "350ba631e875380dc4dd596777e870a0dddd26c1d2ea5247e84eed020a77e950  /PowerShell.Linux.x64.7.4.20.nupkg" | sha256sum -c - \
    && mkdir --parents /usr/share/powershell \
    && dotnet tool install --add-source / --tool-path /usr/share/powershell --version 7.4.20 PowerShell.Linux.x64 \
    && dotnet nuget locals all --clear \
    && rm /PowerShell.Linux.x64.7.4.20.nupkg \
    && ln -s /usr/share/powershell/pwsh /usr/bin/pwsh \
    && chmod 755 /usr/share/powershell/pwsh \
    && chmod 755 /usr/share/powershell/.store/powershell.linux.x64/7.4.20/powershell.linux.x64/7.4.20/tools/*/any/pwsh \
    && find /usr/share/powershell -print | grep -i '.*[.]nupkg$' | xargs rm \
    && test "$(dotnet --version)" = 8.0.425

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-numpy python3-pip python3-scipy \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir gurobipy==13.0.0

LABEL org.opencontainers.image.title="FrontierCS2 verifier metaopt_combined_te_gap_search" \
      dev.frontier.problem-id="metaopt_combined_te_gap_search"

COPY judge /opt/frontier/task
COPY repo /opt/frontier/task/pristine
COPY .frontier-generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY .frontier-generated/verifier/verifier.json /opt/frontier/verifier.json
COPY .frontier-generated/effective-task.toml /opt/frontier/effective-task.toml
# The NuGet cache the C# project builds against offline: restored from the project's pinned
# package versions, then checked package by package (NuGet's own .nupkg.sha512) against the exact
# 40-package set of the old images; any other or extra package fails the build.
RUN cp -r /opt/frontier/task/pristine/project /tmp/restore \
    && dotnet restore /tmp/restore/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj \
    && rm -rf /tmp/restore \
    && (cd /opt/frontier/nuget && find . -name '*.nupkg.sha512' | sort | while read -r f; do printf '%s  %s\n' "$(cat "$f")" "${f#./}"; done) > /tmp/nuget-got \
    && printf '%s\n' \
         'L5XqXbZfuyniJVrrGO0Dog9XDothon7X5cT2zEGze5+ZD7OjKdS8CFjMeqJTlt3cnPlEmBmarUEJF7iCwnp2LQ==  decisiondiagrams/1.3.0/decisiondiagrams.1.3.0.nupkg.sha512' \
         'Xm7ZnxNkJPSyRvVLCvKm/meCm8tXkudTWlLrAL8CccqUWwU+7iorhrISc4sDwFuXH+/nanzNCS+kZssQ5brMBA==  google.ortools.runtime.linux-arm64/9.12.4544/google.ortools.runtime.linux-arm64.9.12.4544.nupkg.sha512' \
         'CEgaRlLYLo02tyqOgN4BLD1F6JviWJlA094HXu2EOl2uyZn21kfNFGPvMzF1Lg4/hPsDfAeGjFnQZnrlgVSyTQ==  google.ortools.runtime.linux-x64/9.12.4544/google.ortools.runtime.linux-x64.9.12.4544.nupkg.sha512' \
         'Z7/sXdDIfDHBvpK796U5ObM/i+MlBwCTC+x79c+QO1diVk3twrKtNuDixTUDoc6mGVjnT5/KpCx/i9eM+ILCXQ==  google.ortools.runtime.osx-arm64/9.12.4544/google.ortools.runtime.osx-arm64.9.12.4544.nupkg.sha512' \
         'z5Tv0O2CnAQ1IfRNPPJhxGm0fEoT7G7mLVMMuTSxk63pUB1bvZjqeZLYMt9kagNVFENpRcoASndxHZ6MPhvXZw==  google.ortools.runtime.osx-x64/9.12.4544/google.ortools.runtime.osx-x64.9.12.4544.nupkg.sha512' \
         'CdBvueWjdMtOd/bp3UVm4ZG2EH5Vp4VHMmv04GEEynOfDsKpp056awlGj62n6oTGUFUJNvJZ+sP0o7rg14CF+A==  google.ortools.runtime.win-x64/9.12.4544/google.ortools.runtime.win-x64.9.12.4544.nupkg.sha512' \
         '3urR/0d3AaqyiWstb/8t2kZwao1VgxLM/zV17BsA3HsSI03QjrPXO1CczkfcDEEsd3E7AQ6WA7BkfaxdtCkhEA==  google.ortools/9.12.4544/google.ortools.9.12.4544.nupkg.sha512' \
         'M6t0YsS6TrSoRWDVTGdWmOfDBMwdCs48iXuoAtwI5qdVI5YXk3NGYAEcQ4h1oA6ib4vPo/mf8JIrrAPu8dHT1w==  google.protobuf/3.29.3/google.protobuf.3.29.3.nupkg.sha512' \
         'PsQSTMIFALRxn2k8AeK/jDtiogJSkfbubuIjlGHxiPso03+auEcyPYkbszbZjb/evVVbDWmGxuhF7YM6qVDQ7Q==  gurobi.optimizer/11.0.3/gurobi.optimizer.11.0.3.nupkg.sha512' \
         'LJaYhRX5VxTUuD9WUPGD3GpWTgs89SVfoOPvSEdt66tL3lQvny9sR/ZiC3px1qUV5EFebS44i2CBeiliHVaQ3w==  microsoft.csharp/4.7.0/microsoft.csharp.4.7.0.nupkg.sha512' \
         'a/iSwnRZb+LHFk49hQOyThh/ZNC3vsbZsF65XwQIb863qF6msmhdQtxGXFL28Ob2NsCz/drEj28BJd/YPpLRBg==  microsoft.netcore.platforms/1.1.0/microsoft.netcore.platforms.1.1.0.nupkg.sha512' \
         'HvAzpoaIqrmZfsHAN4rLFji0r7YY5TP8r3SdkziXN7qU9KCpRIG+zfcBx+mIri/jkBNqjq4iWIfuYNtFBjSQ/g==  microsoft.netcore.targets/1.1.0/microsoft.netcore.targets.1.1.0.nupkg.sha512' \
         'YPSAvGBn08PCDx7SsmUPtwIRZ/0GvhdS0cpL6O6dg1Bn1xOims7v46SgunnQq6vmSd3JS6KUzKdgSZyGZbptjw==  microsoft.z3/4.11.0/microsoft.z3.4.11.0.nupkg.sha512' \
         'mbJSvHfRxfX3tR/U6n1WU+mWHXswYc+SB/hkOpx8yZZe68hNZGfymJu0cjsaJEkVzCMqePiU6LdIyogqfIn7kg==  newtonsoft.json/13.0.3/newtonsoft.json.13.0.3.nupkg.sha512' \
         'NgZfRYct4n6hCld5ZUPgM1eFFVUtRDcBYOvif2tIz8IlFY5tjUsUduIJx3ydyQIqOPC9eJKoBKAcTWB8blIeDw==  nlog/5.2.8/nlog.5.2.8.nupkg.sha512' \
         '0Z49CpYxM008jdEB/j2/uEPCeqoSxYm5igEb6n11f8oBT8THf+gf1vnjzqKCNIfL/7aYbLPv4v3Jicy1cHWOBw==  quikgraph/2.3.0/quikgraph.2.3.0.nupkg.sha512' \
         'OYYZdiCF/b3K/AzGWQ2PLQH5BnQRzQfHbbUgucMpJqBe9jsPTnyQh92oMabffFcTcBR2EOpJyMcp9D05fs8yvA==  stylecop.analyzers/1.1.118/stylecop.analyzers.1.1.118.nupkg.sha512' \
         '8ISvyTlddLTyUsR7fQ43jmdta4tgM6aGNrZItYgF43ct0i/x3tBdPIyFU9LnaFspt1P+HLtaMz8Bir5kIqPr+g==  system.collections.immutable/7.0.0/system.collections.immutable.7.0.0.nupkg.sha512' \
         'ynuVLTDaFIfKTkOqUigXte4m5+EFNwYoEBEvxnp1EnZsOdQC85S7BCbREIu8+bu2Tpzh9a9zbvIVpRo15V8FGw==  system.collections/4.3.0/system.collections.4.3.0.nupkg.sha512' \
         'bFj+HjYY5/h2hMHOp+/H07Gb19+NJTX54ntixS9EHxG2eyEiXWvNYvQJ4CwqFiMcTbGb4zuPq1ubClyGYN2rJA==  system.diagnostics.debug/4.3.0/system.diagnostics.debug.4.3.0.nupkg.sha512' \
         'VERv7pT0MsuP047BDJKah7MHp28VKi6doRupnEHOsPZZE88hiUSZDw4SLU+FiUUJHpgGyEwCha2h/Mk5M30w6g==  system.dynamic.runtime/4.3.0/system.dynamic.runtime.4.3.0.nupkg.sha512' \
         'gj0rowjLBztAoxRuzM0Nn9exYVrD+++xb3PYc+QR/YHDvch98gbT3H4vFMnNU6r8poSjVwwlRxKAqtqN6AXs4g==  system.globalization/4.3.0/system.globalization.4.3.0.nupkg.sha512' \
         'v8paIePhmGuXZbE9xvvNb4uJ5ME4OFXR1+8la/G/L1GIl2nbU2WFnddgb79kVK3U2us7q1aZT/uY/R0D/ovB5g==  system.io/4.3.0/system.io.4.3.0.nupkg.sha512' \
         'YbkO+a5vd5+8intkg+6PVEnN0FyBsFI19wRH5lanOyqrfDQXhLmZ91MjdHRKcuLDpc0TgA6iNBf6wyzPrlzebQ==  system.linq.expressions/4.3.0/system.linq.expressions.4.3.0.nupkg.sha512' \
         '6sx/4exSb0BfW6DmcfYW0OW+nBgo1UOp4vjGXfQJnWsupKn6LNrk80sXDcNxQvYOJn4TfKOfNQKB7XDS3GIEWA==  system.linq/4.3.0/system.linq.4.3.0.nupkg.sha512' \
         'QJvKPSE5vR0APHEUALotteV2u1TVk6pUHsNXbnsgKbYBWascWyxOc4kmexuV682MLwZNxuH1Pmk6rLFzfwZhIw==  system.objectmodel/4.3.0/system.objectmodel.4.3.0.nupkg.sha512' \
         '6b5fYr9ksZR6SYVzNzBqXQmAaGtY1mWYnpQAarBKp+C79NhUPRtX1bs4B5BS8nXzObcwVKc1fk+jVyCKCshdaQ==  system.reflection.emit.ilgeneration/4.3.0/system.reflection.emit.ilgeneration.4.3.0.nupkg.sha512' \
         'rVivBylr0ISQegifkgJvo4mLdk651qB8lBS1UKg6xgRW8yo0Enwpu5OpYz+we6n9go97QaMdzl/wGafPGrKUNQ==  system.reflection.emit.lightweight/4.3.0/system.reflection.emit.lightweight.4.3.0.nupkg.sha512' \
         'vkUFFGejarllQQ8RKkdfuBUQpVlTR9HMDEawKOBDajOSGN08Bz8EjC0zi2fcE7RXQikLbEb1WYJQP3So8mmIGA==  system.reflection.emit/4.3.0/system.reflection.emit.4.3.0.nupkg.sha512' \
         'Bs/ZksjX/Zq2QyqwK+mBoBtlWChabiangloGTU78zgjZ5zRPA/oZsDOiRZ1CsLgOjBQAzjm0ehdShpq4glsEdQ==  system.reflection.extensions/4.3.0/system.reflection.extensions.4.3.0.nupkg.sha512' \
         '1LnMkF9aXKuQAgYzjoiQaL9mwY7oY6KdaO/zzeLMynNBEqKoUfLi5TiKIewoAF+hkxfGTZsjkjsF1jRL4uSeqg==  system.reflection.primitives/4.3.0/system.reflection.primitives.4.3.0.nupkg.sha512' \
         'aK6BpjW5ryrun8j8j+faA1bvTaTrMvgaift1YTuWcU6PGh9MEr0NM177sDQIzHp0QxSDfxNWTV+yYsonIFVnfw==  system.reflection.typeextensions/4.3.0/system.reflection.typeextensions.4.3.0.nupkg.sha512' \
         'IyW2ftYNzgMCgHBk8lQiy+G3+ydbU5tE+6PEqM5JJvIdeFKaXDSzHAPYDREPe6zpr5WJ1Fcma+rAFCIAV6+DMw==  system.reflection/4.3.0/system.reflection.4.3.0.nupkg.sha512' \
         'kGfbKPHEjQj8Uq1Apgj4jBStkRJkZ0Hdr0Jv3+aL7WGrAZVLF5Rh5h0Yc3FgDB5uXDbHiJk/WhBaZPVwKmuB1A==  system.resources.resourcemanager/4.3.0/system.resources.resourcemanager.4.3.0.nupkg.sha512' \
         'aAoysZwr1QJvhoeqU4KupPQytPAy+L3imfrLYYxW1XNpre9/fMjmCtgq48EuXdUHckkTY7+ARMd4d4YopmBbvA==  system.runtime.extensions/4.3.0/system.runtime.extensions.4.3.0.nupkg.sha512' \
         'kqsiSfCAc8+v3Ez719s21lGthxuNi6lhAGmCGH3jdL9KMK+T8V9zsFrzQ/enDL1ISwTWRlcFh2Nq5yFx6wcU+w==  system.runtime/4.3.0/system.runtime.4.3.0.nupkg.sha512' \
         'b/f+7HMTpxIfeV7H03bkuHKMFylCGfr9/U6gePnfFFW0aF8LOWLDgQCY6V1oWUqDksC3mdNuyChM1vy9TP4sZw==  system.text.encoding/4.3.0/system.text.encoding.4.3.0.nupkg.sha512' \
         'fUiP+CyyCjs872OA8trl6p97qma/da1xGq3h4zAbJZk8zyaU4zyEfqW5vbkP80xG/Nimun1vlWBboMEk7XxdEw==  system.threading.tasks/4.3.0/system.threading.tasks.4.3.0.nupkg.sha512' \
         'l6J1G9zmn6r5xU+DSp/Vxgx6eG+qUvQgdpgo28m1gEwfNyG6HqlF6h2ESDXZCYEPnngsmkTQ+q7MyyMMTNlaiA==  system.threading/4.3.0/system.threading.4.3.0.nupkg.sha512' \
         'zzc0Cf/3pMFwqntYpqzr366bYOzDmCTvjZvT0TlcqPp3Wa4urf2NjOMT8SsRVD5JqyynSR9Y6/37q6nKUvMeQw==  zenlib/3.1.6/zenlib.3.1.6.nupkg.sha512' \
       > /tmp/nuget-expected \
    && diff /tmp/nuget-expected /tmp/nuget-got \
    && rm /tmp/nuget-got /tmp/nuget-expected \
    && chmod -R a+rX,a-w /opt/frontier/nuget

# .NET named mutexes use /tmp/.dotnet even when TMPDIR is set; the root filesystem is read-only.
RUN rm -rf /tmp/.dotnet && ln -s /frontier/tmp/.dotnet /tmp/.dotnet \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/effective-task.toml \
    && chmod 0555 /opt/frontier \
    && chmod -R a+rX,a-w /opt/frontier/task

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
