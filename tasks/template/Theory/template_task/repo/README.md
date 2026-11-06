# Count Primes

Make `count_primes` in `src/primes.c` as fast as you can while keeping it exact.

## Task
`uint64_t count_primes(uint32_t n)` (declared in `src/primes.h`) returns the number of primes
`p` with `2 <= p <= n`. The starter implementation uses trial division: it is correct and slow.

## What you may change
- Only `src/primes.c`. The header, the bench program (`bench/main.c`) and the `Makefile` are
  fixed; the judge builds from its own copies.
- The function must compute the answer: tables of precomputed counts, reading files, spawning
  threads or processes, and detecting the judge's inputs are forbidden.

## Interface
`src/primes.h` declares `uint64_t count_primes(uint32_t n);`. `bench/bench N` prints
`count_primes(N)` on one line, e.g. `./bench/bench 10000000` prints `664579`.

## Correctness
The result must be exact for every `n` from 0 to 100,000,000. A build failure, or any run that
prints a wrong count, crashes or exceeds 60 seconds, scores the whole submission 0.

## Workload
One workload is one value of `n`, run as one fresh process of `bench/bench`. Train measures
three public workloads: `n` = 5,000,000, 10,000,000 and 20,000,000. Final measures those three
again plus three held-out workloads from the same range. Each workload is run five times for
your build and five times for the judge's reference build, interleaved; its time is the
smallest of its five wall times.

## Scoring

```text
score = clamp(50 + 25 * log2(S), 0, 100)
S     = geometric mean over workloads of (reference time / your time)
```

50 at the reference's speed, 75 at twice its speed, 100 at four times.

**Beats the reference**: on every workload, wall_seconds (the smallest of five measured wall times
of a fresh bench process) must be strictly less than the reference's (zero tolerance; equality
fails), with every run printing the exact count.

## Running locally

```sh
make                     # cc -O2 -std=c11 -Wall -Wextra -Isrc -o bench/bench bench/main.c src/primes.c
./bench/bench 10000000   # prints 664579
```

## Environment
The judge uses the same compiler and flags on a 2-CPU machine like yours.
