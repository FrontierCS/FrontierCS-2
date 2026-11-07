# Exact Sequence Alignment

Implement the removed body of `obtainAlignmentHirschberg` in `edlib/src/edlib.cpp` so that the
retained Edlib library and its FASTA aligner reconstruct exact optimal alignment paths as fast
as possible. A submission is valid only if every distance, endpoint and path is exact; the
score rises as the measured path-run time falls.

## Task
Edlib (Šošić and Šikić, Bioinformatics 2017) computes edit distance (insertions, deletions and
substitutions, each costing one) between a query and a target, in three modes: NW (global),
SHW (prefix) and HW (infix). Distance and location search are retained; the path
reconstruction core has been removed, so the provided tree fails whenever a path is needed and
is not a performance baseline. Inputs are DNA strings, from short boundary cases to
30,000-base aligned cores at several mutation rates. Any optimal tied path is acceptable.

The reference is a private, complete implementation of the removed body (not the one that was
removed: it reconstructs faster and with less storage). It is restored in the same source tree
and measured afresh in every evaluation; no naive-baseline score is published.

## What you may change
- Only the body of `obtainAlignmentHirschberg` in `edlib/src/edlib.cpp`; local helper types and
  lambdas may live inside it. Keep the result's allocation ownership and return-status
  conventions.
- Everything else is fixed. Outside the body the file must keep the provided text, ignoring only
  blank-line count, trailing spaces/tabs and CRLF vs LF.
- The body may not close its function (nothing at namespace scope); line continuations,
  trigraphs and raw string literals are refused.
- Forbidden in the body: preprocessor directives and the calls `clock`, `asm`.
- No code may run outside a call of this function: the built program must keep the reference
  build's constructor/destructor tables and runtime symbol definitions.

## Interface
The existing static declaration of `obtainAlignmentHirschberg` in `edlib/src/edlib.cpp` is
binding. Its inputs are the encoded query and target arrays, their reversed arrays, the equality
definition and the known optimal score. On success return `EDLIB_STATUS_OK` with an allocated
alignment array and its length, keeping the surrounding library's allocator ownership
convention. Operation bytes: 0 match, 1 query insertion, 2 query deletion, 3 mismatch. The
public `edlibAlign` API and its header describe each result field and empty-input behaviour.

Example: query `ACGT`, target `ACCT`, global mode: distance 1, one valid path `0,0,3,0`,
extended CIGAR `2=1X1=`. A FASTA file:

```text
>query
ACGT
```

The judge runs the built `build/bin/edlib-aligner` with two command lines (SHW or HW replace NW
on those rows):

```text
build/bin/edlib-aligner -m NW -p -f CIG_EXT -r 1 -k -1 query.fa target.fa   # path run
build/bin/edlib-aligner -m NW -l -r 1 -k -1 query.fa target.fa              # location-only run
```

The location-only run does the same start-up, input, distance and location search and prints
the distance and locations, but never calls the reconstruction. A private API adapter also
checks locations, equality pairs and output modes the CLI does not fully expose.

## Correctness
Pre-measurement rejections score 0: an edit outside the body or a refused body form is
`out_of_scope`; a forbidden call or directive, or a changed launch surface, is `content_policy`.

Any other failure gives no score: a build, process or resource failure; an incomplete suite; any
run whose complete output differs from the retained program's format; a wrong distance or
endpoint; a path not consuming the query and chosen target interval at exactly the optimal cost
under the configured equality; location-run locations differing from the reference's; a
`Cpu time of searching` line missing, malformed, not positive or above the measured time.

Mandatory short cases cover all modes and DISTANCE/LOC/PATH requests, empty inputs, equality
pairs, bounded and unlimited searches, word boundaries, ties, repetitive strings and unequal
lengths; final adds an unseen batch of random short cases.

The function computes each alignment from its inputs and returns; it may not use files,
environment, processes, threads or timing, end the process, or print. Runs are pinned to one
CPU; IPC, sockets, signals and process creation are blocked. Binaries are dynamic x86-64 ELF;
each process has at most 900 seconds.

## Workload
30 public workloads: nominal core lengths 10,000 and 30,000 × similarities 60%, 70%, 80%, 90%
and 99% × modes NW, SHW and HW, one query/target pair each. Train uses these 30; final freshly
measures them plus 30 separately seeded held-out pairs from the same distribution. Mutations are
substitutions, insertions and deletions; the labels are generator strata, not certified
distances or the paper's datasets. SHW adds a target suffix; HW adds prefix and suffix flanks.
Public seeds and hashes are in `challenge_data/public_workloads.json`; private seeds are not
distributed. Recognizing fixture identities or changing behaviour by phase is not permitted.

Timing: each time is the judge-measured elapsed wall time of a whole process (monotonic clock,
start to OS-reported exit, scheduling noise included). Per workload and implementation there
are fifteen fresh path runs and fifteen fresh location-only runs in fifteen interleaved rounds
alternating candidate/reference order; no batching, caching, extra sample discard or
retry-to-pass. What remains in the reconstruction time (below) is the reconstruction plus
conversion and printing of its path.

## Scoring
```text
score = 100 * mean_w( A_w / (A_w + T_w) )
T_w   = median elapsed time (s) of the fifteen candidate path runs on workload w
A_w   = (nominal_core_length_w / 100000)^2 seconds   (a challenge constant)
```

The score uses only the candidate's absolute path-run times; any correctness failure means no
score. A workload at T_w = A_w contributes 50; T_w → 0 approaches 100.

**Beats the reference**, decided per workload after every correctness requirement passes, by
exactly two metrics that must both hold on every workload (no compensation; RSS is not
compared):
- `cpu_time_seconds`: smallest path-run time minus smallest location-only time of the same
  executable on the same input; positive and strictly less than the reference's (no tolerance;
  equality fails).
- `core_peak_live_bytes`: heap (peak live bytes from the malloc family, C++ new and mmap) +
  stack (deepest extent below the function's frame) + all writable/thread-local static storage
  of `edlib.cpp`, in the outermost reconstruction call, the maximum of five untimed instrumented
  path runs; no greater than the reference's (equality passes). Moving storage between heap,
  stack and static does not lower it; a stack deeper than 4 MiB or memory obtained another way
  (program break, remapping, shared memory, raw system calls, the dynamic loader) invalidates
  the evaluation.

## Running locally
```sh
cmake -S /workspace/repo -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
python3 challenge_data/generate.py public-10000-60-NW /tmp/w   # writes query.fa and target.fa
build/bin/edlib-aligner -m NW -p -f CIG_EXT -r 1 -k -1 /tmp/w/query.fa /tmp/w/target.fa
```

Workload IDs are listed in `challenge_data/public_workloads.json`.

## Environment
The judge's aligner runs on one pinned CPU with OMP/BLAS threads set to one; compilation is
serial. No GPU is used.
