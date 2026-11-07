# Fixed-Length Clustered Sparse Multiplication

Reconstruct and speed up the fixed-length clustered sparse matrix multiplication (SpGEMM) path of
the clusterwise-spgemm repository. The output must match the source's ordered clustered
arithmetic exactly (within the stated tolerance). The score rewards speed over the reference on
60 CPUs; beating the reference needs a strict 15 % speedup on every workload.

## Task

The retained program `sample/ReorderedFlengthClusterSpGEMM.cpp` multiplies a row-reordered
sparse matrix A by B, with A's rows grouped into clusters of eight that are processed together.
The clustered representation and multiplication (the paper's contribution) are removed; the
private reference is the original upstream contribution. Source: PASSIONLab/clusterwise-spgemm at
`a6f1f8bd3d85f3c4b7a47f22b4220ee7386d09b8` (arXiv 2507.21253v1, "Improving SpGEMM Performance
Through Matrix Reordering and Cluster-wise Computation").

## What you may change

- `CSR_FlengthCluster.h`, `BIN_FlengthCluster.h`, `hash_mult_flengthcluster.h`, `hash_mult.h`,
  `cluster_utility.h`, `CSR_VlengthCluster.h`, `BIN.h`, and your own helpers below
  `frontier_solution/`; at most 24 changed files; no file may be deleted.
- Everything else (the main program, dependency sources and public adapters) is source-locked.

The program is built as two translation units: `sample/ReorderedFlengthClusterSpGEMM.cpp`
(`main`, loading, reordering, CSR construction, flop count, output; no editable header) and
`sample/ReorderedFlengthClusterSpGEMM_kernel.cpp`, which includes the editable headers and defines
the fixed entry point `frontier_clustered_spgemm` (it builds `CSR_FlengthCluster` from A and runs
`HashSpGEMMCluster`). Editable-header macros and declarations never reach the main program. The
kernel object may not define any symbol that the main object, another linked object or a linked
shared library defines or references (for example an explicit specialization of a locked
template member), apart from the entry point, discarded template/inline copies, the replaceable
global `operator new`/`operator delete`, and inline code the unmodified headers already emit.

## Interface

The retained `CSR_FlengthCluster<IT, NT>` fields have the following meanings.
These are representation requirements, independent of the multiplication method:

- `cluster_sz` is the fixed width, eight.
- `csr_rows` is the padded scalar row count, a multiple of eight.
- `rows` is the number of eight-row groups: `csr_rows == rows * 8`.
- `cols` is the scalar column count; the padded square output requires
  `cols == csr_rows`.
- `nnzc` is the total number of stored column entries across all groups, not
  the number of scalar lane values.
- `rowptr` contains `rows + 1` entries, begins at zero, is nondecreasing and
  ends at `nnzc`. Group `g` stores entries in `[rowptr[g], rowptr[g + 1])`.
- `colids` contains `nnzc` scalar column indices in `[0, cols)`, with no
  duplicates within a group. Their order within a group may vary; all-zero columns may be
  omitted, and a missing value counts as zero.
- `values` contains `nnzc * 8` binary64 values. Entry `j` stores lane `k`
  at `values[j * 8 + k]`, representing scalar row `g * 8 + k` in its group.

Populate every output metadata field and keep the arrays valid until the
caller consumes the result; inconsistent metadata is malformed output. The
public benchmark does not validate the output, so check this interface in your
own tests.

## Correctness

The result follows the source's ordered clustered arithmetic. For each group of eight reordered
A rows, visit the union of their stored column indices in ascending order (missing lanes are
zero); for each visited k, traverse the stored columns of B[k,:] in ascending order. The first
structural path to an output column sets each lane to A_lane[k] * B[k,column] (no threshold);
later paths update a lane only when abs(A_lane[k]) >= `double(float(0.000001))` =
`9.999999974752427e-7`, as product plus existing value.

Every supplied value and every expected nonzero value is checked with tolerance
`1e-9 + 1e-8 * abs(expected)`.
The capture must be complete, finite and at most 7,516,192,768 bytes, and every timed process
must compute its own result.

Each of these, caused by your build or runs, scores 0 for the submission: a broken rule above
(rejected), a compile, link or symbol-check failure, compilation over 600 s, wrong or malformed
output, a native process that crashes, exits nonzero, or exceeds 180 s wall, 10,800 CPU s or
48 GiB address space, total measured memory over 56 GiB, the parent's own memory over 8 GiB, or
the workspace over 32 GiB. Running out of the overall
evaluation time, a reference failure and judge faults fail the evaluation without a score.

## Workload

One workload is one whole sparse matrix with a frozen identity and a structural RCM row ordering;
cluster width is always eight. The matrices come from the SuiteSparse Matrix Collection (CC-BY-4.0)
and cover structural mechanics, numerical stencils, irregular systems and directed graphs.

The authentic input reader transforms a numeric entry `x` to binary64 `x + 1`;
a pattern entry becomes `1`. Symmetric off-diagonal entries are mirrored.
Stored positions remain present even when their transformed value is zero.
Only the rows of A are permuted. B retains the original matrix, so a reordered
case does not apply the same permutation to B. Empty rows and columns pad the
square shape to a multiple of eight.

| suite | workloads |
| --- | --- |
| development (train) | 8 matrices from development parents |
| final | 8 matrices from distinct final parents |

Matrix identities, values and permutations are private; the suites share no parent.

Each workload has seven alternating candidate/reference pairs, each in a fresh
process. Parent timing begins before input access and includes loading,
reordering, representation construction, multiplication, complete output
transfer and process exit. There is no untimed same-input warmup. Output is
validated before the next process begins; child timers or correctness claims
do not determine the result. Native processes cannot access private judge files,
network services or other processes, or change their CPU affinity.

## Scoring

```text
G     = geometric_mean over workloads of (reference_time / candidate_time)   (median times)
score = 100 * G / (1 + G)
```

Reference parity scores 50; the score approaches but never reaches 100.

**Beats the reference** when, on every workload, the candidate median execution time < 0.85 × the reference
median, and derived throughput > reference throughput / 0.85 (exactly 15 % does not pass).
Throughput counts twice the original stored scalar paths, including stored zeros, divided by
elapsed milliseconds and 1000; it is a reciprocal of the same time. If the reference's timings on any workload exceed 5 % relative
median-absolute-deviation or 5 % odd/even median drift, the evaluation fails without a score.

## Running locally

Run `./frontier-build.sh` (which compiles and links the two translation units
with the Verifier's flags; it does not run the Verifier's symbol checks), then
`./frontier-bench.sh /absolute/matrix.mtx /absolute/order.txt 8` from your writable
repository. Each ordering file lists each original row index exactly once,
zero-based, in the desired A row order. Public debugging uses two logical CPUs
with a 16-GiB Agent budget. It does not measure the scientific 60-CPU target.

## Environment

The judge runs on 60 logical CPUs with 64 GiB of memory and a 32-GiB workspace; CPU affinity stays
within a parent-selected mask, and OpenMP uses the configured thread count with dynamic
adjustment disabled. The internal evaluation deadline is 60 seconds shorter than the outer limit.
