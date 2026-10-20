# MetaOpt Combined-TE Gap Search

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read `/frontier/task/repo/README.md` for the problem and
`/frontier/task/bin/README.md` for the testbed lifecycle and quotas.

Prepare `/workspace/project` with `/frontier/task/bin/prepare_workspace.sh`.
Edit `MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs` and optional C#
helpers below `MetaOptimize/TrafficEngineering/Candidate/`. Other project files
are locked. Build offline with:

```sh
dotnet build /workspace/project/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj -c Release -p:RestoreSources=/opt/frontier/nuget -p:NuGetAudit=false
```

Like every task, this one is submitted as a patch. `prepare_workspace.sh`
gives `/workspace/project` a Git baseline equal to the read-only starter
`/frontier/task/repo/project`. `make_submission.sh` stages only the editable
surface and writes `/workspace/payload/solution.patch`, a text diff with paths
under `project/`; package it separately:

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
frontier-submit train /workspace/train.tar
```

Train, store-final and explicit final use the same uncompressed POSIX tar
containing exactly `payload/solution.patch` (an explicit `payload/` directory
member is also accepted); the complete tar is limited to 4 MiB. The Verifier
applies the patch to its own copy of the starter project. The patch may edit
`project/MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs` and edit or
add text files under `project/MetaOptimize/TrafficEngineering/Candidate/`;
context must match the starter exactly, and binary, rename, mode-change and
deletion records are rejected. The resulting project is limited to 96 files and
2,000,000 source bytes.

Use early validated train feedback, then train again after substantive progress.
There are **100 train calls** for accepted submissions; failed accepted verification also consumes
one. Train covers the original public fifteen cases. Final freshly evaluates
those same public cases plus fifteen hidden cases, and both score and PASS cover
all thirty. No train measurement is reused for final.

Train feedback includes `mean_gap` (the candidate's absolute arithmetic mean
gap), `instance_count` (the number of evaluated public cases), and
`beat_spec_progress.gap`. Progress contains `passed`, `failed`, `total`,
`pass_percent` and `fail_percent`; percentages are rounded to two decimal places.
Invalid candidates receive zero score and all-failed progress without unmeasured
mean or count fields. Per-case identities, decisions, reference values and margins
remain private. Final reports its reference verdict only to the operator.

Save your selected final while continuing to work:

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/final.tar payload/solution.patch
frontier-submit store-final /workspace/final.tar
```

A successful stored receipt replaces the selected archive without ending work.
After train feedback, decide whether to rebuild and save a new selection. Local
file edits alone do not update the stored archive. The last train is never
selected automatically. An accepted `frontier-submit final /workspace/final.tar`
locks that archive and overrides any stored selection; stop platform calls and
exit after acceptance.

Formal answering is **18,000 seconds (5 hours)**. Independent final receives
3,600 seconds after collection and cleanup.
