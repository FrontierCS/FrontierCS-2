# MetaOpt Combined-TE Gap Search

Write a budgeted search that finds sparse, bounded traffic matrices on which a combined
traffic-engineering heuristic loses as much throughput as possible against the optimum. The score
rises with the mean independently replayed gap. **First priority: reach at least the reference
search's gap on every instance (see Scoring); the continuous score is secondary** and never
compensates for an instance below the reference.

## Task

The project is a pruned, buildable C# subtree of Microsoft MetaOpt (Namyar et al., NSDI 2024).
The combined heuristic runs POP and demand pinning and keeps the better throughput, so a useful
matrix must hurt both. Implement the search in the real `MetaOptimize/TrafficEngineering`
module. The judge replays every returned matrix with exact linear programs; claimed objective
values are ignored.

## What you may change

- `project/MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs`.
- New or changed C# source files (`.cs`) under `project/MetaOptimize/TrafficEngineering/Candidate/`.
- Every other project file is fixed. Symlinks, deletions, binary files, executable
  files, a project of more than 96 files or more than 2,000,000 bytes are rejected. Build products,
  `.git` and Python bytecode are not submitted.
- No recognizing particular instances (topologies, capacities, seeds) or the train/final phase to
  return precomputed matrices; the search must work from the instance it is given.

## Interface

```csharp
public static int[] Find(AdversarialSearchInstance instance, IGapOracle oracle)
```

Return one level index per `instance.Pairs` entry, in the same order. `Find` runs once per
instance in a fresh process; its stdout and stderr are discarded. Each instance discloses its
directed edges and capacities, its eligible ordered source-destination pairs, each pair's fixed
POP partition and up to four paths per pair (a path is an array of edge indices). `Levels` is the
demand quantization list; `SearchBlocks` partitions the pair indices into three-pair blocks
(using them is optional).

```csharp
GapEvaluation result = oracle.Evaluate(levelIndices);   // Gap, OptimalThroughput, PopThroughput, DemandPinningThroughput
```

At most `instance.QueryBudget` (256) successful calls are allowed; an invalid matrix throws an
argument exception and consumes no query, and repeated matrices consume repeated queries. The
returned final matrix is replayed without using the budget. You may evaluate matrices yourself
from the disclosed instance (your own LP or model); that work is limited only by the process
deadline. The reference has the same instance information and no more.

## Correctness

Every instance has 27 eligible pairs, levels `[0, 6, 18]`, nine search blocks, at most four
paths per pair and three POP partitions. A returned array must have exactly 27 integer indices
in `[0, 2]`, with at most nine selecting a nonzero level. Demand pinning sends every positive
demand of at most 6 on its first (shortest) path; the pinned load on every edge must be at most
its capacity within `1e-7`, otherwise the matrix is invalid. All numbers must be finite. The
project must build within 240 seconds and each instance process must finish within 30 seconds, using at most 64 processes or threads, 64
open files and 120 CPU-seconds.
A missing, invalid, crashing, timing-out or non-building submission scores zero on every case.

## Workload

Fifteen public instances for train; final replays those fifteen with their unchanged seeds plus
fifteen hidden ones (thirty). Each set holds five capacity perturbations of each directed edge
skeleton of the SWAN, Abilene and B4 (TEAVAR) topologies, with integer capacities 24–36. Every call receives its complete instance; nothing is withheld from `Find`. For a demand
vector `d` the judge solves path-form fractional multicommodity-flow LPs (each pair's total path
flow at most its demand, each edge's total flow at most its capacity):

- `OPT(d)`: maximum total flow with full capacities.
- `POP(d)`: each of the three partitions optimizes its assigned demands on a copy with one third
  of every capacity; the three throughputs are summed (MetaOpt's `PopEncoder`).
- `DP(d)`: every demand of at most 6 is carried in full on its shortest path and removed from
  capacity; demands above 6 are then optimized over their paths (`DirectDemandPinningEncoder`).
- `COMBINED(d) = max(POP(d), DP(d))` (`TECombineHeuristicsEncoder`), and
  `gap(d) = OPT(d) − COMBINED(d)`.

## Scoring

```text
G = mean over the selected instances of gap(d)        # 15 in train, 30 in final
score = clip(100 * (1 - exp(-G / 40)), 0, 100)        # score_unbounded = G
```

G = 0 scores 0, G = 40 about 63.2, and the score approaches 100 as G grows.

**Beats the reference** when, on every selected instance (15 in train, 30 in final), the replayed
`gap` is at least the reference's. Tolerance: none; ties pass, and there is no compensation across
instances. The reference is MetaOpt's unclustered
`MaximizeOptimalityGap` (the primal-dual rewrite of `OPT − max(POP, DP)` as one joint MILP,
solved with Gurobi at a deterministic six-work-unit budget and four threads, validated with one
oracle call); it is not in the project.

## Running locally

```sh
dotnet build /workspace/repo/project/MetaOptimize.Challenge/MetaOptimize.Challenge.csproj -c Release -p:RestoreSources=/opt/frontier/nuget -p:NuGetAudit=false
```

## Environment

.NET 8 SDK 8.0.425 with an offline NuGet cache at `/opt/frontier/nuget`; 4 CPUs.
