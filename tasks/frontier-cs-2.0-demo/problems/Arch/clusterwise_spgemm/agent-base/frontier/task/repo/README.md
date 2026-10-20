# Fixed-Length Clustered Sparse Multiplication

## Answering and Final Evaluation
The answering window is 18,000 seconds (5 hours), including preparation,
editing, and waits for development submissions. At most 3 train calls may
consume a cumulative 21,600 seconds within the remaining answering window;
a train call cannot extend that window. Independent final evaluation begins
after answering and Agent cleanup and has its own 7,200-second budget.
The final budget does not consume or extend the answering window.
All matrices, seven candidate/reference pairs per workload, CPU/memory limits,
correctness rules and scientific score thresholds remain unchanged.

## Problem
Reconstruct and improve the fixed-length clustered sparse multiplication path
in the supplied C++17 repository. The retained program is
`sample/ReorderedFlengthClusterSpGEMM.cpp`; its clustered representation and
multiplication interfaces are the task. Implement the missing contribution and
any supporting code within the permitted patch surface.

The Verifier uses whole sparse matrices from distinct development and final
parents, covering structural mechanics, numerical stencils, irregular systems
and directed graphs. Each matrix has a frozen identity and structural RCM row
ordering. Width is always eight. Matrix identities, values and permutations are
private. You can create small Matrix Market inputs for local debugging.

## Numeric Contract
The authentic input reader transforms a numeric entry `x` to binary64 `x + 1`;
a pattern entry becomes `1`. Symmetric off-diagonal entries are mirrored.
Stored positions remain present even when their transformed value is zero.
Only the rows of A are permuted. B retains the original matrix, so a reordered
case does not apply the same permutation to B. Empty rows and columns pad the
square shape to a multiple of eight.

The result follows the source's ordered clustered arithmetic, including a
special first-insertion rule. For each group of eight reordered A rows, visit
the union of their stored column indices in ascending order. Missing lanes have
value zero. For each visited index k, traverse the stored columns of B[k,:]
in ascending order. On the first structural path to an output column, initialize
each lane to A_lane[k] * B[k,column], without thresholding. On later paths,
update a lane only when abs(A_lane[k]) is at least
`double(float(0.000001))`, exactly `9.999999974752427e-7`. The update is the
product plus the existing value, in that order. Cluster membership can therefore
affect values; ordinary unqualified A@B or a filtered product with a posthoc
correction is not the general correctness contract.

Return the complete numeric result through the retained clustered output
interface. Column order may vary, but column indices must be in range and
unique within a group. You may omit or include all-zero columns. Missing values
are treated as zero. Every supplied value and every expected nonzero value is
checked with tolerance `1e-9 + 1e-8 * abs(expected)`. Nonfinite values,
significant missing or extra values, malformed output, and incomplete captures
are rejected. The complete capture must fit 7,516,192,768 bytes. Every timed
process must compute its own valid result.

## Clustered Output Interface

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
  duplicates within a group. Their order within a group may vary.
- `values` contains `nnzc * 8` binary64 values. Entry `j` stores lane `k`
  at `values[j * 8 + k]`, representing scalar row `g * 8 + k` in its group.

Populate every output metadata field and keep the arrays valid until the
caller consumes the result. A numerical result with inconsistent metadata is
malformed output. The public benchmark reports execution and is not the
Verifier's complete numeric or output-shape validator; check this interface
when constructing local tests.

## Editable Surface And Debugging
Modify only `CSR_FlengthCluster.h`, `BIN_FlengthCluster.h`,
`hash_mult_flengthcluster.h`, `hash_mult.h`, `cluster_utility.h`,
`CSR_VlengthCluster.h`, and `BIN.h`. You may add your own helpers below
`frontier_solution/`. The patch is limited to 24 changed files and 262144 bytes.
The main, remaining dependency sources and public adapters are source-locked.
The private reference restores the original upstream contribution unchanged.

The retained program is built as two separately compiled translation units,
linked with the main object first. `sample/ReorderedFlengthClusterSpGEMM.cpp`
holds `main`, argument handling, matrix loading, reordering, CSR construction,
the flop count and output; it includes no editable header.
`sample/ReorderedFlengthClusterSpGEMM_kernel.cpp` includes the editable headers
in the original order and defines the fixed entry point
`frontier_clustered_spgemm` (declared in
`sample/ReorderedFlengthClusterSpGEMM_kernel.h`), which constructs
`CSR_FlengthCluster` from A, runs `HashSpGEMMCluster` and hands the product to
the main program. Macros, overloads and declarations in editable headers
therefore never reach the main program. A build is rejected if the kernel
object defines a symbol that the main object or another linked object or
archive defines or references (an explicit specialization of a locked template
member, for example), other than the entry point and template or inline
instantiations whose copy the linker discards in favour of the main object's,
or a symbol a linked shared library exports, other than the replaceable global
`operator new`/`operator delete` forms and inline code the unmodified headers
already emit.

Run `./frontier-build.sh` (which compiles and links the two translation units
with the Verifier's flags; it does not run the Verifier's symbol checks), then
`./frontier-bench.sh /absolute/matrix.mtx /absolute/order.txt 8` from your writable
repository. Each ordering file lists each original row index exactly once,
zero-based, in the desired A row order. Public debugging uses two logical CPUs
with a 16-GiB Agent budget. It does not measure the scientific 60-CPU target.

## Measurement And Score
Scientific execution uses 60 logical CPUs and a 64-GiB Verifier with a 32-GiB
workspace. Native children have a 48-GiB address-space ceiling, a 180-second wall
deadline and a 10800 process-CPU-second ceiling. The parent has an 8-GiB memory
ceiling; a valid measurement must keep total measured memory at most 56 GiB.
The shared compilation budget is 600 seconds. Each submission has a 7200-second
outer limit and a 7140-second internal evaluation deadline.
CPU affinity stays within a parent-selected mask;
OpenMP uses the configured thread count with dynamic adjustment disabled.

Each workload has seven alternating candidate/reference pairs, each in a fresh
process. Parent timing begins before input access and includes loading,
reordering, representation construction, multiplication, complete output
transfer and process exit. There is no untimed same-input warmup. Output is
validated before the next process begins; child timers or correctness claims
do not determine the result. Native processes cannot access private judge files,
network services or other processes, or change their CPU affinity.

Both strict conditions must pass on every workload: candidate median execution
time is less than 0.85 times the reference median, and derived throughput is
greater than reference throughput divided by 0.85. Exactly 15% improvement does
not pass. Throughput counts twice the original stored scalar paths, including
stored zeros, divided by elapsed milliseconds and 1000. It is a reciprocal
diagnostic of the same execution time. The continuous score is
`100 * G / (1 + G)` with `G = geometric_mean(reference_time / candidate_time)`:
reference parity scores 50 and the score approaches but never reaches 100. It
does not replace the every-workload conjunction.

Reference timings must pass 5% relative median-absolute-deviation and odd/even
median-drift gates. Invalid correctness, resources, build, security or timing
conditions invalidate the measurement. Development feedback exposes only
candidate aggregate absolute metrics and anonymous per-metric passed/failed
counts and percentages. Reference values, individual workload rows and all
final measurements remain private.

## Attribution
The source is PASSIONLab/clusterwise-spgemm at
`a6f1f8bd3d85f3c4b7a47f22b4220ee7386d09b8`, accompanying arXiv 2507.21253v1,
Improving SpGEMM Performance Through Matrix Reordering and Cluster-wise Computation.
Retained source and dependency notices accompany the repository. Matrix inputs
come from the SuiteSparse Matrix Collection under its CC-BY-4.0 declaration;
each private dataset retains the complete original matrix headers, attribution,
citation metadata and a disclosure of the runtime transformations described here.

## Development Feedback Fields
Only development submissions expose these candidate-only aggregate fields:

- `candidate_median_execution_time_ms`: median absolute execution milliseconds.
- `candidate_median_throughput_mflops`: median absolute derived MFLOPS.
- `candidate_max_output_bytes`: largest captured candidate product in bytes.

These are absolute candidate statistics, without reference values or workload identifiers. Final measurements remain private. Anonymous per-metric pass/fail counts and percentages retain the existing feedback rules.
