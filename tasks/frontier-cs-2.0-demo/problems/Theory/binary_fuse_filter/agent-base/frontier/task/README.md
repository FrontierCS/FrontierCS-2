# Binary Fuse Filter Construction and Lookup

Implement and optimize static approximate-membership filters for distinct `uint64_t`
keys. This task preserves Graf and Lemire's segmented three-wise binary fuse
filter API, with 8-bit and 16-bit fingerprints. The population routines have been
removed from `include/binaryfusefilter.h`. Allocation, lookup and serialization
remain available. Alternate XOR population routines are also removed to avoid
shipping an alternate construction answer; XOR performance is not scored.

Copy the read-only repository to your workspace:

```sh
/frontier/task/bin/prepare_workspace.sh /workspace/binary_fuse_filter
cd /workspace/binary_fuse_filter
```

You may edit `include/` and add header-only helper files there, up to 32 changed
files. Preserve the `binary_fuse8_*` and `binary_fuse16_*` allocation, populate,
contain and free APIs. The filter structs may change. Helpers must build with
`cc -std=c99 -O3 -fno-lto -DNDEBUG` and `-lm`. Do not change the trusted
benchmark, compiler, scoring, or submission tools. The unchanged `make query`
entrypoint is exercised. Its XOR sections can report construction failures when
the optional XOR stubs remain; both Binary Fuse sections must run. The original
tests are retained as development tools, but tests requiring XOR construction
will fail until those optional routines are implemented.

The static filter must remain usable after the caller discards its input keys.
It must support multiple independent instances and release all of its storage.
Keep mutable state in the filter object and heap allocations made through standard
`malloc`, `calloc`, `realloc`, and `free`; the evaluator accounts for these allocations.
Do not retain mutable global state, bypass allocation accounting, access judge
files, detect verification mode, or substitute output for API execution.

Correctness requires successful construction and no false negatives for **every**
inserted key, including small edge cardinalities. Timed sets have 10,000, 100,000
and 1,000,000 distinct keys, both fingerprint widths, and three families:
SplitMix-style permuted indices, odd-stride arithmetic sequences, and keys with
varying high 32 bits. Input order is shuffled. The original query mixtures contain
0%, alternating 50%, and 100% hits; an additional mixture has exactly 25% hits
with keys and truth labels shuffled together. Each absent stream is disjoint
from the inserted set by construction, with up to
1,048,576 queries. Development uses seed 20261007. Final includes the complete
development suite and a second, private seed.

False positives remain a validity gate and are also an explicit quality metric.
For `m` absent
queries and `b` fingerprint bits the maximum count is
`ceil(m/2^b + sqrt(2*(m/2^b)*log(1200/1e-6)) + (2/3)*log(1200/1e-6))`.
This Bernstein/union-bound allowance assumes independent Bernoulli false-positive
events at nominal probability `2^-b`. Deterministic hashing does not prove that
assumption; this is a reproducible empirical test, not a universal FPR guarantee.
The added 25%-hit stream uses the same formula with `400/4e-8` inside the
logarithm. Original stream thresholds are unchanged. At most 1,152 old gates
and 384 new gates in a full final keep the combined nominal family error bound
below `1e-6`, under the stated model.

For each workload, five paired repetitions compare your implementation with the
exact upstream reference on the same CPU. Execution order alternates. The three
original metrics are retained allocation bytes including the filter object (maximum over
repetitions), construction nanoseconds (median), and nanoseconds per lookup
(median for each query mixture). Allocation tracking excludes allocator metadata
and temporary construction storage that is freed before lookup. Queries and
input keys are generated outside the timed regions. The lookup loop includes
writing one byte per answer. CPU frequency, caches and scheduling still add noise.

Two additional metrics are effective false-positive probability and space overhead
relative to the theoretical minimum. Each construction measures `epsilon = k/m`
on eight disjoint absent-query blocks outside lookup timing: the original
1,048,576-key absent block plus seven new blocks. Repeated executions are not
pooled as independent observations. A conservative Bernoulli KL/Chernoff interval
inverts `m*D(k/m || p) <= log(2/alpha)` with `alpha = 1e-6/360`. Each workload
uses the worst point value across five constructions, with the maxima of their
lower and upper interval bounds. The same filter-quality observation is reused
across its four query mixtures, including the all-hit mixture.

For each construction, space overhead is
`100*((8*resident_bytes/n)/log2(1/epsilon)-1)` percent. Its interval is the monotone
transform of the epsilon interval using that construction's actual bytes.
Zero or all false positives give an observed epsilon of 0 or 1, but an **undefined**
space point; no pseudocount, nominal epsilon, finite floor or clamping is used.
Such a row cannot clear the space predicate. Invalid counts/non-finite inputs
are rejected. Interval endpoint limits are bounds, not substituted point values.

Both new metrics are lower-is-better and are judged as certified non-inferiority
guards against preregistered, reference-only anchors, not as strict improvements.
For each workload the effective-FPR anchor is the larger of the fingerprint's
nominal rate `2^-b` and the worst point value across at least five complete
reference-only rounds; the space-overhead anchor is the worst reference-only
point value. A workload clears a guard when the candidate's certified **lower**
interval bound does not exceed its anchor, with defined finite points and bounds.
A candidate is rejected only when the data certify, at the family-wise level
above, that its true rate or overhead exceeds the anchor; overlapping intervals
are never read as equal true values. The anchors were fixed from reference-only
data before any successor solver result, stay private, and never relax the three
strict performance comparisons. Resolution is bounded by the sample: with
8,388,608 absent queries the guard cannot certify an excess below roughly
1.6-1.7 times the nominal rate at 16-bit fingerprints or about 1.04 times at
8-bit, so smaller true differences are accepted, not endorsed.

The descriptive score is the arithmetic mean of `100*R/(C+R)` over every workload
and the original three metrics,
where `C` and `R` are candidate and reference measurements. A valid restoration
is around 50 points. The separate **strict PASS** requires `C < R` for every
original metric plus both non-inferiority guards on every workload, in addition
to all correctness gates. A high aggregate
score cannot compensate for any failed comparison. Train feedback reports only
your aggregate absolute metrics and the pass/fail counts for each metric; exact
reference values, preregistered anchors and per-workload decisions stay private.

Build an uncompressed POSIX tar using:

```sh
/frontier/task/bin/make_submission.sh /workspace/binary_fuse_filter /workspace/answer.tar
frontier-submit train /workspace/answer.tar
frontier-submit store-final /workspace/answer.tar
frontier-submit final /workspace/answer.tar
```

Train and final use the same layout: the sole regular file is
`payload/solution.patch`, a UTF-8 unified Git patch against the supplied repository.
An optional `payload/` directory entry is allowed. No executable bit is required
for the patch. The patch is at most 524,288 bytes and the complete tar at most
1,048,576 bytes. Links, devices and binary patches are not accepted. Helpers use
`/workspace`; never write below `/frontier/task`.

You have **100 train calls**, a **12,000-second cumulative train pool**, and
**18,000 seconds total answering time**, including development evaluations.
Each verification and the independent final evaluation have a 180-second timeout.
The train pool can be exhausted before the call count; it does not promise 100
maximum-length calls. Final has its own clock after answering. Agent and Verifier
each use 2 CPU and 4 GiB memory; the Agent workspace is 4 GiB. No GPU is needed.

`frontier-submit train` returns immediately with a verification ID; retrieve that
development result later with `frontier-submit status <verification_id>`.
You must save the final answer you select with
`frontier-submit store-final /workspace/answer.tar` and keep its accepted receipt.
You decide which answer to save and whether to replace that selection. Saving
does not evaluate the answer or consume a train call. There is no save-count
quota; `remaining_checkpoint_submissions` is `null`. Payload, audit and
Agent-time limits still apply. Editing local files does not change saved bytes.
An accepted explicit final is irreversible and takes precedence over the saved
selection.

Development feedback contains `beat_spec_progress`, with `passed`, `failed`,
`total`, `pass_percent`, and `fail_percent` for each of the five metrics.
`solver_visible_metrics` contains these candidate-only summaries:

- `mean_resident_bytes`: Candidate-only mean of per-workload maximum retained allocation bytes.
- `mean_build_ns`: Candidate-only mean of per-workload median construction nanoseconds.
- `mean_lookup_ns`: Candidate-only mean of per-workload median nanoseconds per lookup.
- `mean_effective_fpr`: Candidate-only mean of per-workload worst empirical false-positive probability.
- `space_overhead_defined_workloads`: Count of candidate workloads with a defined interior-domain space-overhead point.

These summaries disclose no reference values, preregistered anchors or workload identities.
