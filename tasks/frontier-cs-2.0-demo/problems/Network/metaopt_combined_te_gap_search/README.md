# MetaOpt Combined-TE Gap Search

The task statement is at `/frontier/task/repo/README.md`.

The read-only source tree is at `/frontier/task/repo` (the C# project is `project/`). Copy it
into your workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `project/MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs` there and add any C#
helpers under `project/MetaOptimize/TrafficEngineering/Candidate/`; nothing else may change. Build offline against the baked NuGet cache:

```sh
dotnet build /workspace/repo/project/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj -c Release -p:RestoreSources=/opt/frontier/nuget -p:NuGetAudit=false
```

The .NET 8 SDK (8.0.425) and PowerShell (`pwsh`) are installed. Nothing outside `/workspace`
is writable.
