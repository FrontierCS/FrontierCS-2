# Binary Fuse Filter Construction and Lookup

Implement and optimize static binary fuse filters for distinct `uint64_t` keys: construction must
succeed with no false negatives, and the score compares your memory, construction time and lookup
time with the exact upstream filters on the same CPU. **First priority: beat the reference on every
workload under all five comparisons (see Scoring); the descriptive score is secondary** and never
compensates for a failed comparison.

## Task

This is Graf and Lemire's segmented three-wise binary fuse filter library
(`FastFilter/xor_singleheader`; Binary Fuse Filters, ACM JEA 2022) with 8-bit and 16-bit
fingerprints. The population routines of `include/binaryfusefilter.h` have been removed;
allocation, lookup and serialization remain. The alternate XOR population routines are removed
too; XOR performance is not scored.

## What you may change

- `include/`: edit the headers and add header-only helpers there, at most 32 changed files.
- Keep the `binary_fuse8_*` and `binary_fuse16_*` allocate, populate, contain and free APIs;
  the filter structs may change. Helpers must build with `cc -std=c99 -O3 -fno-lto -DNDEBUG`
  and `-lm`.
- Everything else (benchmarks, tests, build files) stays unchanged; no symlinks or binary files.
- The filter must stay usable after the caller discards its keys, support independent instances
  and free all of its storage. Keep mutable state in the filter object and allocate only through
  standard `malloc`, `calloc`, `realloc` and `free` (they are accounted). No mutable global
  state, no bypassing allocation accounting, no access to judge files, no detecting the
  train/final role, no substituting output for API execution, and no recognizing particular
  workloads (seeds, key families or generated key/query sequences) to change behaviour. The filter
  code must not write to standard output (the harness reports through it).

## Interface

The judge compiles its own harness and a bridge against your `include/`:
`cc -std=c99 -O3 -fno-lto -DNDEBUG -I include -c bridge.c`, a separate harness translation unit,
linked with `-lm`. It calls the public API for each workload: allocate for `n` keys, populate
from a shuffled key array, query, free. The unchanged entrypoint is also run: `make -j1 query
CC=cc` then `./query`; its output must contain both `binary_fuse8:` and `binary_fuse16:`
sections (its XOR sections may report construction failures while the optional XOR stubs
remain). The retained tests are development tools; those needing XOR construction fail until
those optional routines exist.

## Correctness

Construction must succeed and every inserted key must be found, on every workload and on the edge
cardinalities `n ∈ {2, 3, 128, 1025}` (each family, both widths). Freed storage must be released
(no leaked bytes). False positives are also a gate: for `m` absent queries and `b` fingerprint
bits at most `ceil(m/2^b + sqrt(2·(m/2^b)·log(1200/1e-6)) + (2/3)·log(1200/1e-6))`, and for the
25%-hit stream the same formula with `400/4e-8` in the logarithm (a Bernstein/union-bound
allowance under independent Bernoulli false positives at rate `2^-b`; an empirical test, not a
guarantee). Each judge command runs with a 30-second limit, 4 GiB address space, 30 CPU seconds
and 8 MiB of output. Any failure scores zero.

## Workload

A workload is one fingerprint width (8 or 16), key count (10,000, 100,000 or 1,000,000), key
family (SplitMix-style permuted indices, odd-stride arithmetic sequences, keys with varying high
32 bits) and query mixture: 0%, alternating 50% and 100% hits over 2^20 queries, plus
exactly 25% hits with keys and truth labels shuffled together. Input order is shuffled and absent
keys are disjoint from the inserted set. Train uses seed 20261007 (72 workloads); final adds a
second, private seed (144). Each workload runs five paired repetitions, candidate and reference
alternating, on the same CPU. Measured: retained allocation bytes including the filter object
(maximum over repetitions; temporary construction storage freed before lookup is excluded),
construction nanoseconds (median; allocate plus populate) and nanoseconds per lookup (median per mixture, one byte
written per answer). Keys and queries are generated outside the timed regions.

## Scoring

```text
score = mean over workloads w and m in {resident_bytes, build_ns, lookup_ns} of 100 * R_w,m / (C_w,m + R_w,m)
```

`C` and `R` are candidate and reference measurements; matching the reference scores about 50.

**Beats the reference** when every correctness gate passes and, on every workload (72 in train,
144 in final), `resident_bytes`, `build_ns` and `lookup_ns` are each strictly lower (`C < R`, no
tolerance) and both certified non-inferiority guards hold (tolerance: the private reference-only
anchor and the interval below):

- **Effective FPR** `ε = k/m`, measured per construction on eight disjoint absent blocks (the
  2^20-query block plus seven more), outside lookup timing.
- **Space overhead** `100·((8·resident_bytes/n) / log2(1/ε) − 1)` percent, undefined when `ε` is
  0 or 1 (an undefined point never clears).

The ε interval is a Bernoulli KL/Chernoff interval `m·D(k/m || p) <= log(2/α)` with
`α = 1e-6/360`; the space interval is the ε interval mapped through the space formula with that
construction's bytes. Each guard uses the maximum point, lower and upper bound over the five
constructions. A workload clears a guard when its certified
lower bound does not exceed a fixed anchor derived only from reference runs: for FPR the larger
of `2^-b` and the worst reference point, for space the worst reference point. Anchors are
private.

## Running locally

```sh
cd /workspace/repo && make query && ./query
make test        # retained unit tests (XOR tests fail until XOR population exists)
```
