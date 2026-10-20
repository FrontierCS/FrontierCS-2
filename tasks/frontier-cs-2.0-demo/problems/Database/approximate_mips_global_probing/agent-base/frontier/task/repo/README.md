# FARGO paper-aligned query core

Implement an efficient approximate maximum inner product search query core in
`src/alg.cpp`. The starting program enumerates buckets in numeric order and never
uses adaptive stopping. The index, input vectors, result heap, exact inner
products, duplicate suppression, and measurements are supplied by the harness.

The research target is FARGO's global multi-probing query algorithm, rather than
a single threshold callback: construct a query-local probing frontier, choose
and advance probes across all hash tables, and decide when further verification
is unnecessary. The reference follows the published paper's squared quantization
distance and adaptive termination formula. The upstream implementation has
different formulas; the paper is the mathematical specification for this task.

Paper: [FARGO: Fast Maximum Inner Product Search via Global Multi-Probing](https://www.vldb.org/pvldb/vol16/p1100-zheng.pdf),
PVLDB 16(5), 1100–1112, particularly Sections 3.2.2–3.2.3 and 4.2,
Equations 11–12 and 16, and Algorithm 1. Random XBOX transformation and norm
partitioning are supplied; this is a query-core task, not full index construction.

## Interface

Implement the factory declared in `src/query_core.h`. A fresh `QueryCore` is
created for each query. `reset(L,K,p)` supplies the normalized Gaussian projections
of the query, in table-major order. `next(out)` emits a `(table,key,score)` bucket
descriptor. The key's highest bit corresponds to projection zero. A positive
projection has hash bit one; zero and negative projections have bit zero.
The reported score must equal the sum of **squared** projections at disagreeing
bits. Arbitrary probe orders are allowed, but if fully consumed, the sequence must
contain every bucket exactly once. The harness checks validity independently.

`stop(state)` is called **before** consuming the next bucket. It receives the
current kth inner product, result count, next bucket's QD, approximation ratio,
partition maximum possible inner product, table count, and calibration path.
Never stop with fewer than k results or a nonpositive kth result. The harness
replays the generated query-global sequence for each norm partition, verifies
each candidate once, and keeps exactly the requested k results. A deterministic
norm upper bound can terminate the partition loop. Small partitions are scanned.

The interface supports `1 <= K <= 20`; functional tests exercise varied small
values, tied/zero projections and reset reuse. Performance tests use K=12 and L=5.
State is query-local, except immutable calibration caching. No dataset IDs, ground
truth or result buffers are exposed to the core.

## Mathematical specification (available offline)

The external paper link is attribution, not a required network dependency.
For completeness, the paper's adaptive rule is specified here. Let `I_k` be the
current kth inner product, `c` the approximation ratio, `U` the partition bound
`M_i * ||q||`, `d` the next bucket's squared QD, and `L` the number of tables.
When `I_k >= c * U`, that partition can stop deterministically. Otherwise set
`theta = acos(I_k / (c * U))`. The supplied CDF represents
`phi(d; theta) = Pr[QD <= d]` for the paper's transformed angular model.
Equation 16 defines `alpha_1 = 1 - phi(d; theta)` and
`alpha_2 = 1 - (1 - alpha_1)^L = 1 - phi(d; theta)^L`.
Its adaptive condition is `alpha_2 < p_tau`; the reference uses `p_tau = 0.1`.
This is the paper's probability of at least one table's bucket remaining unseen,
not `(1 - phi)^L` and not an expected remaining-candidate count. It assumes
independent tables; the numerical CDF approximation has the limits stated below.
For k results, substitute the current kth result for the paper's best result.

The paper's global probing rule chooses the lowest-QD unvisited bucket across
all tables. A bucket's QD is the sum of squared normalized projections whose
bits differ from the query's natural key (Equations 11–12). Implementing an
efficient frontier, avoiding duplicate bucket generation, and applying the
termination rule are part of the task. Alternative valid algorithms remain
allowed; the reference code and measured reference metrics stay private.

## Calibration resource

`calibration_path` is a read-only, dataset-independent numerical CDF table for
the paper's squared-QD distribution. All integers and floats are little-endian.
The layout is: 8 bytes `FARGOSQ1`; int32 K, rows, columns; float64 theta step and
score step; then `rows*columns` float64 CDF entries in row-major order. Theta starts
at zero and covers pi/2; QD starts at zero. Entry (r,j) is a conservative-bin
approximation at theta `r*theta_step`, QD `j*score_step`. Individual nonzero squared
contributions are rounded **up** before K-fold convolution. Rounding theta up
and QD down gives conservative lookup directions. Floating-point quadrature is
validated numerically; this is not a formal interval-arithmetic certificate.

## Workload and score

Evaluation uses two original-paper datasets: Audio (54,387 x 192) and MNIST
(60,000 x 784). Each has a fixed random sample of 300 in-dataset queries and
disjoint train/held-out halves. Each half uses a different fixed hash seed.
Train covers both datasets; final independently includes both train and held-out
halves of both datasets. MNIST uses flattened original grayscale pixels, with no
labels or normalization. Every query is tested at c={0.6,0.8,0.9} and k={1,10,50,100}. Unlike the old
stopping-only problem, the result heap has exactly k entries. Ground truth is
exhaustive top-100, computed outside candidate execution.

Per operating point, quality is `sqrt(recall * overall_ratio)` with both metrics
clipped to [0,1]. Overall ratio is the paper's mean rankwise inner-product ratio;
the kth ratio is diagnostic only. Efficiency is the geometric combination of
paired-baseline candidate-work speedup (weight .75) and query-time speedup
(weight .25), without speedup clipping. The point score is
`100 * quality * efficiency / (1 + efficiency)`; total score gives equal weight to all 24 train or 48 final
operating points (12 per dataset/split). This scalar benchmark convention is **not** a formula from the paper.
Lower candidate work does not imply lower wall-clock latency. Individual quality,
work and latency measurements are retained in the validation evidence.

Only `src/alg.cpp` may change. Do not inspect process roles, hidden files, workload
identities, or generate metric output. Submit a unified source patch using
`bash make_submission.sh` . Build a public API
smoke example with `make && ./fg`. Invalid, incomplete, duplicate or false-QD probe
streams score zero. Native code must run in the platform's candidate sandbox;
source restrictions are scope checks and are not a security boundary.

## Reference verdict

The primary success criterion is PASS: on every operating point, recall and
overall ratio must each be at least the reference's, the mean number of verified
candidates per query (the harness's exact inner-product count, reported as
`mean_candidates` in aggregate) must be no greater than the reference's, and
measured query latency must be strictly lower. Equality passes the candidate
comparison but not the latency comparison, so recall cannot be bought with extra
verification work. All comparisons are conjunctive; aggregate score cannot
compensate for a failure.
There is no statistically established timing tolerance. The continuous score
remains a separate progress measure with no upper speedup clamp.
