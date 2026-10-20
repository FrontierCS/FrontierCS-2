# EVM State Checkpoints

## Problem
Spectrum is a C++20 smart-contract execution engine, built here from upstream
commit `e67c73cccc48aa76fe78c3d04383cf90b4bbd5b6`. Spectrum concurrently executes
a consensus-ordered stream of Ethereum Virtual Machine (EVM) transactions while
preserving that serial order, and it restores EVM bytecode position and operand
stack from checkpoints after transaction conflicts. The benchmark reads a
colon-delimited workload record and seed, then the artifact's generators produce
ABI-encoded YCSB, SmallBank, or TPC-C-like contract calls continuously for one
second; the separate correctness driver reads a round count and seed and runs
fixed checkpoint traces. The benchmark writes transaction statistics to standard
error, while the correctness driver writes checkpoint timing, retained-memory,
live-checkpoint, and checksum fields to standard output. A correct submission
must restore the saved code iterator, stack height, and every 256-bit stack item,
preserve every other live checkpoint, reclaim tracked storage, and complete each
benchmark with finite positive committed throughput. The objective is to
maximize the returned 0–100 score, which combines committed transactions per
second, checkpoint creation and restoration latency, and bytes retained while
checkpoints are live. The checkpoint-capable stack implementation associated
with the paper’s operation-level rollback work has been removed; supply its
replacement in the permitted header while retaining the required C++ interface
and behavior.

## Design space
All submitted code must reside in the one editable header. Within that boundary,
the definitions of `StackTop` and `Checkpoint`, their private state, and the
bodies needed for construction, stack access, mutation, copying, moving,
destruction, checkpoint creation, and checkpoint restoration are editable.
Public declarations used by the existing Spectrum sources and the correctness
driver remain a compatibility boundary: the constructors, public `height`, stack
operations, checkpoint copy and move support, and `ExecutionState` checkpoint
operations described under `## Interface` must continue to compile and satisfy
their stated semantics. Types and call sites outside the header, EVM instruction
semantics, transaction processing, workload generation, statistics formatting,
and the build configuration are fixed and are not part of the editable surface.

## Workload
There is no downloaded or file-backed dataset and no fixed transaction-record
count. The YCSB, SmallBank, and TPC-C-like generators from the Spectrum artifact
construct ABI-encoded calls in memory from a workload record and a pseudorandom
seed, using the bundled contract bytecode. Each performance process generates
and executes calls continuously for one requested second, so the number of
records depends on achieved throughput.

The development suite contains two cases:

```text
YCSB:10000:0.9       seed 1907
Smallbank:10000:0.9  seed 3253
```

For YCSB, `10000` is the key-domain size and `0.9` is the Zipf exponent.
Each call has 11 distinct 256-bit arguments: ten keys followed by one value.
The contract reads keys in argument positions 1, 3, 5, 7, and 9 and writes keys
in positions 2, 4, 6, 8, and 10. For SmallBank, `10000` is the account-domain
size and `0.9` is the Zipf exponent. The generator selects one of six contract
calls—balance, deposit checking, write check, transact saving, send payment, or
amalgamate—and supplies one to three generated 256-bit arguments, as required by
that call.

The graded suite has different fixed seeds and three cases:

```text
YCSB:15000:0.9
Smallbank:15000:0.9
TPCC:100:5
```

The YCSB and SmallBank fields have the meanings above. In `TPCC:100:5`, `100`
is the item-domain parameter given to the artifact’s uniform generator and `5`
is the number of order lines placed in a NewOrder call. This generator fixes the
warehouse and district identifiers to zero and emits NewOrder, Payment, and
Delivery calls. It is not a 100-warehouse workload. Graded seeds are not
disclosed.

The workload record is colon-delimited. This is a literal development record,
with its fields labeled:

```text
YCSB:10000:0.9
  │     │    └─ Zipf exponent
  │     └────── key-domain size
  └──────────── workload name
```

Before performance measurement, the judge also runs deterministic checkpoint
traces against an independent fixed-array oracle. For each suite seed, the
driver performs 400 rounds. Each round pushes 1,000 values, retains 16
simultaneous checkpoints, and performs 24 indexed writes after every checkpoint.
It copy-constructs and move-constructs one selected checkpoint, restores that
object, restores a second original checkpoint, and compares the height and all
1,000 values with the corresponding oracle states. It then erases checkpoints
in a seed-dependent order, destroys the execution state, and requires tracked
live allocation to return to zero. Thus each trace creates 6,400 checkpoints. The first trace per seed retains its original role in the continuous score. Four additional traces per compared variant and seed provide five measurements for the beat-reference timing gate; later trace order rotates between candidate and reference.

Performance normally uses five runs per workload case. The evaluator accepts a
configured repetition count from one through five and takes the median committed
TPS over the selected runs for each case. Development uses the two records and
seeds shown above; grading uses the three graded records, fixed undisclosed
seeds, and the configured repetition count. These generated cases represent the
three workload families; submissions must satisfy the checkpoint contract for
the generated executions and traces rather than recognize the listed records or
seeds.

## Interface
The performance executable reads exactly four positional arguments. The complete
development invocation run by the evaluator is:

```text
<build-tree>/build/bin/bench Spectrum:2:9973:CHECKPOINTED YCSB:10000:0.9 1s 1907
```

The record layout is:

```text
argv[1] = Spectrum:2:9973:CHECKPOINTED
          protocol:executors:table-partitions:EVM-mode
argv[2] = YCSB:10000:0.9
          workload:key-domain:Zipf-exponent
argv[3] = 1s
          duration-with-unit
argv[4] = 1907
          unsigned generator seed
```

Every evaluated invocation fixes the protocol to `Spectrum`, executors to `2`,
table partitions to `9973`, EVM mode to `CHECKPOINTED`, and duration to `1s`.
Only the workload record and its suite seed change. There are no additional
evaluator-supplied command-line flags.

The program writes a multiline statistics record to standard error. The field
consumed by the evaluator has this literal layout:

```text
commit        62965.0000 tx/s
│             │          └─ unit
│             └──────────── decimal committed throughput
└────────────────────────── field name
```

The line must start with `commit`, contain a finite positive decimal value, and
end with `tx/s`. The surrounding record contains timestamp, duration, memory,
execution, operation, latency-bucket, and latency-percentile lines; those fields
do not enter the score.

The correctness executable is compiled and invoked as follows for every suite
seed:

```text
g++ -O3 -DNDEBUG -std=c++20 -I<submitted-tree> -I<judge-headers> \
    <correctness-driver>.cpp -o <correctness-driver>
<correctness-driver> 400 <seed>
```

On success, it writes one space-separated record to standard output. For
example:

```text
246 65 13408 16 206159111842
 │   │    │   │       └─ checksum
 │   │    │   └───────── live_checkpoints
 │   │    └───────────── retained_bytes_per_checkpoint
 │   └────────────────── restore_nanoseconds_per_checkpoint
 └────────────────────── create_nanoseconds_per_checkpoint
```

Timing values vary by machine. The driver validates restored state before
printing and requires `live_checkpoints` to be 16. The checksum summarizes a
value from each restored trace and does not enter the score.

The following C++ interface and semantics are required:

- `StackTop(uint256* begin, uint256* end)` constructs an empty stack associated
  with the supplied stack-space bounds; default construction must remain valid.
- Public `StackTop::height` is the current number of items. `push(value)` adds an
  item, `pop()` removes and returns the former top, and `top()` returns the
  current top.
- `get(index)` and `operator[](index)` return a const reference to the item
  `index` positions below the top, so index zero denotes the top.
- `get_mut(index)` returns a mutable reference to that same item. References
  returned by `pop()`, `top()`, `get()`, `get_mut()`, and `operator[]` must remain
  usable by the existing EVM instruction call sites across subsequent `pop()`
  and `get_mut()` calls in the same instruction. `push(value)` must also work
  when `value` refers to an item in that same stack.
- `ExecutionState::save_checkpoint()` returns a copyable and movable
  `Checkpoint` containing the current `position.value().code_it` and all current
  stack state.
- `ExecutionState::load_checkpoint(const Checkpoint&)` restores the saved code
  iterator, stack height, and every stack value.
- Creating, loading, copying, moving, or destroying one checkpoint must not
  change any other retained checkpoint. Every retained checkpoint remains valid
  until it is destroyed.

For example, this state transition is required:

```cpp
// bottom-to-top stack is [10, 20, 30]
state.position = Position{code + 7};
auto saved = state.save_checkpoint();
state.stack_top.get_mut(1) = 99;  // current stack: [10, 99, 30]
state.position = Position{code + 12};
state.load_checkpoint(saved);     // stack: [10, 20, 30], position: code + 7
```

The complete application is configured and built with:

```text
cmake -S <submitted-tree> -B <submitted-tree>/build -DNDEBUG=ON
cmake --build <submitted-tree>/build --target bench -j 4
```

## Validity
Validity is checked before performance. A submission must be a nonempty textual
Git patch no larger than 65,536 bytes. It must apply cleanly, change exactly the
allowed header, and change no more than one file. Binary content or markers,
absolute or parent-traversing destination paths, and patches without a
`+++ b/<path>` destination are invalid.

Added lines are rejected case-insensitively if they contain any of these tokens:

```text
#define          getenv        system(       popen         fork(
malloc           calloc        realloc       aligned_alloc mmap
sbrk             operator new  operator delete
Statistics       checkpoint_bench             FRONTIER_     int main
```

The submitted header must compile in both the correctness driver and the full
Spectrum application. Every trace must restore the oracle’s height and values,
preserve all other retained checkpoints through copy, move, load, and
destruction, reclaim all tracked storage, exit normally, and print exactly five
integers. Creation and restoration time must each satisfy
`0 < time < 1,000,000,000` nanoseconds. Retained bytes per checkpoint must
satisfy `0 < M <= 67,108,864`, and the live-checkpoint count must be 16.

Allocation tracking begins before the execution state is dynamically allocated.
Requested bytes for that object include all inline members; live nested
`new`/`new[]` allocations and the vector storage for 16 checkpoint objects are
also included. A live allocation reachable by several objects is counted once.
At the point where all 16 checkpoints are live, peak live requested bytes are
divided by 16 and rounded upward to form `retained_bytes_per_checkpoint`.
Allocation APIs that bypass this tracker are forbidden.

Every performance process must exit normally within 20 seconds, emit the
required `commit` line, and report finite positive TPS. Patch-policy violations,
patch or build failures, correctness mismatches, leaks, malformed profile or
statistics output, timeouts, crashes, and nonpositive TPS all score zero before
performance is considered.

## Scoring
Maximize the returned score. For candidate `C`, workload case `j`, run `r`,
selected repetition count `N`, and case count `J`, define:

```text
t_Cj  = median(t_Cj1, t_Cj2, ..., t_CjN)
T_C   = exp((1 / J) * sum_j(ln(t_Cj)))
C_C   = median creation nanoseconds per checkpoint across suite seeds
R_C   = median restoration nanoseconds per checkpoint across suite seeds
M_C   = maximum retained bytes per live checkpoint across suite seeds
L_C   = sqrt(C_C * R_C)
Q_C   = T_C / (M_C * L_C)^(1/3)
```

Here `t_Cjr` is committed TPS for one run and `T_C` is geometric-mean committed
TPS across cases. The cube-root combination is a challenge constant (not from
the materials). Let `Q_B` be the quality measured in the same evaluation for the
checkpoint backend this task ships — the implementation already present in the
editable header — using the same suite, repetition count, allocation accounting,
latency procedure, build, and runtime settings. The exact scoring rule is:

```text
if not (Q_C > 0 and Q_B > 0):
    unbounded = 0
else:
    unbounded = 100 * log10(Q_C / Q_B)

score = max(0, min(100, unbounded))
```

The evaluator returns `score` as the bounded score and `unbounded` as the
unbounded score; the two differ once `unbounded` leaves `[0, 100]`. Matching the
shipped backend's composite quality is worth exactly 0, a tenfold composite
quality is worth 100, and between those two points the score increases strictly
with `Q_C`. A quality below the shipped backend reports a negative unbounded
value and a bounded score of 0.

## Calibration
The checkpoint backend already present in the editable header is the zero of
this scale: a submission whose composite quality matches it scores exactly 0.00,
and each of the four axes — committed throughput, creation latency, restoration
latency, and retained bytes — has to be earned from there. A patch that fails a
validity gate also scores zero, so a low score is not by itself distinguishable
from a format, build, or correctness failure by its number; the judge message
does distinguish them, reporting a scored submission as a valid checkpoint
backend with its score and a gate failure as the specific violation.

Differences on this scale are quality ratios: `S` points is a factor of
`10^(S/100)` in composite quality, so 2 points is 4.7% and 10 points is 26%.
These are one-second CPU measurements. Treat changes smaller than 4 score points
as timing noise unless they repeat; a repeatable change of at least 4 score
points is a material improvement. That band is the previously recorded
sub-two-point variance carried onto this scale rather than a fresh measurement,
so it is a guide and not a guarantee.

## Feedback and grading
The judge returns a score and task-approved metrics after every submission. The submission budget is in `/frontier/task/README.md`. **The graded suite
is not the suite you receive feedback on.** Its three workload records are
listed under `## Workload`; its seeds are not disclosed. Development feedback is
computed from the two named development records and their published seeds.
Final grading instead uses all three graded records and their fixed undisclosed
seeds, so the two suites differ in both membership and seeds.

## Resources
The CPU-only environment uses `buildpack-deps:noble` with CMake installed. It
provides 4 vCPUs, 8,192 MiB memory, and 8,192 MiB storage. The configured build
timeout is 1,800 seconds, and the complete evaluation timeout is 3,600 seconds.

The initial dependency configuration and initial build each have a 1,800-second
limit. Each incremental measured build has a 600-second limit. Correctness-driver
compilation has a 60-second limit; each correctness invocation has a 30-second
limit; and each performance invocation has a 20-second limit. The initial
dependency build is cached, and measured builds use that prepared source tree.
