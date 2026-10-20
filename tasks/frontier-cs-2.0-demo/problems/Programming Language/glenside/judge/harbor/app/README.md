# Access-Pattern Rewriting

## Problem
Glenside is a Rust compiler infrastructure project for representing tensor computations and mapping them to specialized matrix hardware; this task uses a retained portion of its upstream source tree. A tensor is a multidimensional array, and the two-dimensional convolutions here compute weighted sums over local windows of input arrays, as used in neural networks. Glenside's intermediate representation describes these computations through access patterns: an outer shape indexes a collection of items, and an item shape describes the array processed at each position. Its `egg` equality-saturation engine stores expressions and their equivalences in an e-graph, whose e-classes group expressions declared to compute the same value. An existing mapping rule recognizes computations supported by a systolic array, a matrix accelerator made of multiply-and-add units.

The input is a suite of 14 records: 12 generated, small convolution expressions with tensor shapes, layouts, padding, strides, and optional surrounding axis operations, followed by two fixed explicit outer-concatenation expressions. The evaluated output is an expression selected from the original root e-class, together with discovery, rewrite-time, and graph-growth measurements; it must preserve the complete output's outer shape, item shape, and numerical values. **Maximize the mean continuous score defined under Scoring, which rewards valid matrix-accelerator discovery and lower rewrite time and e-graph node growth on discovered workloads.** The compositional `MyAnalysis::make` and `MyAnalysis::merge` implementation has been replaced by explicit missing-core stubs; actual accelerator execution is outside this CPU-only task.

## Design space
Implement compositional access-pattern analysis in `code/src/language/solver_extensions/access_analysis.rs`. The complete `egg::Analysis<Language>` implementation is included in its original language-module scope, so the existing types, imports, private fields and helpers remain available. The four generic rewrite constructors in `rewrites.rs` are initially populated with their pinned implementations and remain editable for generic optimization. Rust helpers may be added under `solver_extensions/`. This is a different missing-core boundary within the same access-pattern rewriting task, not a new workload or backend.

The trait signatures, IR and tensor semantics, downstream mappers, rule order, extraction policy, and measurement protocol are fixed. Only pure, compositional analysis and generic semantics-preserving rewrites are permitted. Do not introduce workload lookup tables, I/O, fabricated facts, dedicated convolution lowering, or direct accelerator/opaque-call construction in generic rewrites. Both existing ordinary and blocked systolic-array forms are legal analysis inputs and numerical witnesses. Their pinned mapping implementations remain unchanged; the measured rule schedule ends with trusted registration of the ordinary mapper followed by the unchanged blocked constructor with positive unit tiles `(1, 1)`. Neither mapper is required to fire. These fixed rules are registered outside the editable generic-rule guard; returning a mapper through an editable generic hook does not bypass that guard. There is no required hook activity, source decomposition, node count, or beating witness beyond the published comparisons.

## Workload
There is no downloaded dataset or training set. The evaluator generates two suites of 14 workload records. The first 12 records in each split are byte-equivalent to the original convolution records and use the retained tensor-access convolution constructor. Each split then appends exactly two fixed explicit records: one concatenates two Cartesian products along the first left-outer axis, and one concatenates along the last right-outer axis. Each operand access has rank-two outer and rank-two item shapes with extents from `{2,3}`; item shapes and every nonconcatenated outer dimension match. The concatenation is immediately below `compute dot-product`, with no additional operation or rule family. Generation and numerical seeds are fixed privately and are not part of the public interface. The convolution records use the following dimensions and parameters; these are challenge constants (not from materials), not a full Cartesian product of every listed choice.

| Quantity | Values in the generator |
| --- | --- |
| Batch size `N` | 1, 2, 3 |
| Input channels `C` | 2, 3, 4, 5 |
| Output channels `O` | 2, 3, 5, 6 |
| Input height `H`, width `W` | Each drawn from the integers 6–10 |
| Kernel `(Kh, Kw)` | `(1,1)`, `(2,3)`, `(3,2)`, `(2,2)` |
| Spatial strides `(Sh, Sw)` | Respectively `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)` for those kernel choices |
| Padding | All zero for `(1,1)` kernels; otherwise each side is drawn from 0–2 |
| Layout | Six NCHW and six NHWC cases per suite |
| Dilation and groups | Fixed at `[1,1]` and 1; no bias |
| Surrounding operations | Three cases of each wrapper described under Interface |

NCHW data has shape `[N,C,H,W]`, with OIHW weights `[O,C,Kh,Kw]`. NHWC data has shape `[N,H,W,C]`, with HWIO weights `[Kh,Kw,C,O]`. Here `N` indexes independent inputs, `C` and `O` index input and output channels, and `H`, `W`, `Kh`, and `Kw` are spatial dimensions. Padding order is `[top,left,bottom,right]`; strides are `[height,width]`. Before the wrapper, the output is `[N,O,Ho,Wo]` or `[N,Ho,Wo,O]`, respectively, where:

```text
Ho = floor((H + top + bottom - Kh) / Sh) + 1
Wo = floor((W + left + right - Kw) / Sw) + 1
```

Numerical verification uses two deterministic tensor assignments per workload, containing `f64` values from `{-3/8,-2/8,-1/8,0,1/8,2/8,3/8}`. Rewriting receives symbolic tensor names and shape metadata, not these numerical arrays. No hardware simulator, GPU, or external download is involved. These are representative tensor programs, not identifiers or expressions to recognize; workload order is not part of the constructor interface. The graded run uses the same stated dimensions and repetition protocol, with different generated records and numerical seeds.

## Interface
### Compositional analysis core
The editable `solver_extensions/access_analysis.rs` contains the trait implementation with these signatures:

```rust
impl egg::Analysis<Language> for MyAnalysis {
    type Data = MyAnalysisData;
    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge;
}
```

This is a signature sketch, not a second executable implementation. `make` receives an IR enode whose children identify e-classes with typed facts, plus the `name_to_shape` and `name_to_dtype` symbol environment. Produce the corresponding typed `MyAnalysisData` fact. Access facts contain outer `shape`, `item_shape`, `zero_regions`, `access_pattern_shape_settled`, and `contains_accelerator_calls`. Other variants retain literals, numbers, dtypes, shapes, lists, tuples and operator/layout tags. The fixed language, type definitions and access/range utilities remain in `language.rs`; only the analysis implementation is missing. The public example and native library compile before implementation, but executing analysis intentionally panics until the core is supplied.

`merge` combines equivalent facts in `to` and returns `DidMerge` flags for changes relative to each input. Honor settled shape information, accelerator propagation and recursive tuple merging, including stable repeated rebuilds. Match the pinned behavior rather than implementing aspirational TODOs: in particular upstream's proposed zero-region union in merge is disabled; preserve the existing target zero-region information. Access merges require equal underlying element counts; when both access shapes are settled, both outer and item shapes must agree. Violations panic rather than produce a fact. This is not a general validator of ill-typed programs; do not add blanket rejection requirements for inputs the analysis accepts. The retained typed variants and operator contracts below specify the task IR; non-workload Relay/TVM branches are not deleted or replaced by a surrogate backend.

Required input domain: the typed access-pattern expressions constructed by the convolution and outer-axis-concatenation workloads, their equivalent generic rewrites, and the ordinary/blocked mappings described below. Symbol environments supply tensor shapes; dtype lookup defaults to float32 when absent. Typed literals, shapes, lists, tags, tuples and tuple selection use the retained data variants. Valid axes, shape-compatible products/pairs/concatenations, in-bounds slices, positive window strides and tile sizes, and value-preserving reshapes are caller preconditions, not a demand to validate arbitrary malformed IR. Relay/TVM operators outside this CPU workload domain remain in the source but are not an additional graded input suite.

The concrete rejection boundary used here is the analysis's existing access-merge checks above and operator preconditions, not an unspecified list of unsupported operations. For example, dot-product analysis requires at least one item dimension and yields the same outer shape with an empty item shape; it does not require that the child be syntactically a Cartesian product. Consequently `(compute dot-product (access (access-tensor x) 1))` with `x` shaped `[2,3,4]` produces an access fact with outer `[2]` and item `[]`, not a required panic. This fact alone does not certify that a malformed numerical program is executable. Reshape analysis likewise does not validate element-count preservation; numerical correctness remains independently required. These are existing semantic requirements, now enforced by a private correctness guard; no mandatory hooks or graded records are added.

Analysis must describe the expression, not which rewrite you hope will fire. Follow actual flag propagation: the ordinary and blocked systolic-array analysis branches propagate their children's `contains_accelerator_calls` flags; the presence of such a node does not by itself force that flag to true. The optimization objective remains discovery, time and growth for correct complete outputs. Compact/fused implementations and any semantically correct implementation are legal. Before scored measurements, a private correctness guard checks the documented typed semantics on the supported workload and generic-rewrite domain: outer and item shapes, typed literal/list/tuple/tag values, accelerator propagation, the stated access-merge constraints, target zero-region preservation, and repeated merge/rebuild stability. Claimed zero regions on numerical access values must be sound; complete inference or a particular zero-region representation is not required. Shape-only metadata is not a numerical zero proof. It does not require a particular private dtype encoding for shape facts, or hard-coded DidMerge flags for two unsettled accesses or zero-region-only differences. Ordinary access nodes propagate settledness from their access children; a value-preserving reshape may conservatively retain an unsettled shape, and either representation is legal when its access children are settled; it cannot claim settledness with unsettled access children. Incorrect semantic facts reject the submission without exposing private cases or expected facts; a trusted runtime failure is reported separately. The guard shares the compiled candidate/reference binaries and the existing aggregate 300-second budget; it adds no scored workloads. No reference analysis implementation or precomputed solution facts are included in the public task.

### Generic rewrite constructors
In the rewrite module, `RW` aliases `egg::Rewrite<Language, MyAnalysis>`:

```rust
pub fn flatten_unflatten_any_access() -> RW
pub fn bubble_reshape_through_cartesian_product() -> RW
pub fn bubble_reshape_through_compute_dot_product() -> RW
pub fn bubble_access_concatenate_through_compute_dot_product_not_item_axis() -> RW
```

Each is a zero-argument constructor returning one rewrite, not an expression, tensor, or measurement record:

- `flatten_unflatten_any_access()` supplies the access-level rewrite hook; its equalities must preserve the access's shape and values.
- `bubble_reshape_through_cartesian_product()` supplies the Cartesian-product rewrite hook; its equalities must preserve the product's shape, item pairing, and values.
- `bubble_reshape_through_compute_dot_product()` supplies the dot-product rewrite hook; its equalities must preserve the result's shape and values.
- `bubble_access_concatenate_through_compute_dot_product_not_item_axis()` supplies the outer-axis concatenation hook; its equalities must preserve the concatenated result's shape and values.

Every constructor must be callable even when its rule is inapplicable. Shape-incompatible matches must not panic or equate unequal values. The judge registers these four returned rules in the order above, followed by its trusted copies of `systolic_array()` and `systolic_array_with_blocking(1, 1)`; the supplied analysis trait methods are called by e-graph construction and rebuilding. Legal compact or fused generic implementations remain permitted: no hook must fire and the minimum mapping count remains one.

### IR input and worked numerical output
Worked numerical example (challenge constant, not from materials): data in NCHW shape `[1,1,3,3]` with row-major values `[1,2,3,4,5,6,7,8,9]`, weights in OIHW shape `[1,1,2,2]` with values `[1,0,0,-1]`, unit strides and zero padding produce shape `[1,1,2,2]` with values `[-4,-4,-4,-4]`. Any credited representation must preserve both that output layout and those values; merely finding a matrix node somewhere in the graph is insufficient.

Here is the **original, pre-rewrite expression** for that public case, using the retained convolution constructor's representation:

```lisp
(access-transpose
  (compute dot-product
    (access-cartesian-product
      (access (access-tensor weights) 1)
      (access
        (access-squeeze
          (access-windows
            (access
              (access-pad
                (access-pad (access-tensor data) zero-padding 2 0 0)
                zero-padding 3 0 0)
              1)
            (shape 1 2 2)
            (shape 1 1 1))
          1)
        3)))
  (list 1 0 2 3))
```

Its `MyAnalysis.name_to_shape` environment is:

```json
{"data": [1, 1, 3, 3], "weights": [1, 1, 2, 2]}
```

The input expression is parsed as `egg::RecExpr<Language>` and added to an `egg::EGraph` with `MyAnalysis { name_to_shape, name_to_dtype }`, then rebuilt. `name_to_dtype` is empty for this example and all measured workloads. An access has `(outer shape, item shape)`; its full tensor shape is their concatenation, denoted `outer ++ item`, and `access_axis` is the number of outer dimensions. Axes are zero-based. An empty item shape `[]` denotes a scalar item, not a zero-element dimension. The operators in this input have these contracts:

- `(access-tensor name)` has the named tensor's entire shape as its outer shape and item shape `[]` in `MyAnalysis` and in the graded numerical wrapper. `(access input k)` sets the split after the first `k` full-tensor dimensions without moving values; `0 <= k <= rank`.
- `(access-pad input zero-padding axis before after)` inserts zeros along a full-tensor axis. Both padding amounts are zero here, so the shape and values are unchanged.
- `(access-windows input (shape K...) (shape S...))` requires window and positive-stride vectors with the same rank as the input item shape `I`, with `0 < K[j] <= I[j]`. For outer shape `A`, its output has outer shape `A ++ Q`, item shape `K`, and `Q[j] = floor((I[j]-K[j])/S[j])+1`. It contains each selected window's values.
- `(access-squeeze input axis)` removes a full-tensor dimension of size one, adjusting the split if that dimension was outer. The following `access ... 3` explicitly sets the item boundary used by the product.
- `(access-cartesian-product left right)` requires equal item shapes `I`. With outer shapes `A` and `B`, it produces outer shape `A ++ B` and item shape `[2] ++ I`, pairing one left item with one right item in that order.
- `(compute dot-product input)` takes items shaped `[m] ++ I`, multiplies the `m` corresponding values at each index of `I`, then sums those products over `I`. It preserves the outer shape and returns scalar items `[]`. Here `m=2`.
- `(access-transpose input (list p...))` permutes the full tensor axes in the listed order, retaining the number of outer dimensions. Here it changes the outer order from output-channel/batch/height/width to batch/output-channel/height/width.

The wrapper syntax also includes `(access-insert-axis input axis)`, which inserts a dimension of size one before the specified full-tensor axis. A subsequent squeeze of that inserted axis restores the original access. `(shape ...)` contains dimension sizes, whereas `(list ...)` contains axis indices.

Intermediate **input-expression** shapes make the split concrete:

| Subexpression | Outer shape | Item shape |
| --- | --- | --- |
| Data after `access ... 1` | `[1]` | `[1,3,3]` |
| Windows | `[1,1,2,2]` | `[1,2,2]` |
| Windows after squeeze and `access ... 3` | `[1,2,2]` | `[1,2,2]` |
| Weights after `access ... 1` | `[1]` | `[1,2,2]` |
| Cartesian product | `[1,1,2,2]` | `[2,1,2,2]` |
| Dot product and final transpose | `[1,1,2,2]` | `[]` |

After implementing the missing analysis core, the supplied `frontier-example.rs` adapter checks this original expression and its numerical result. Before implementation it compiles but stops at the explicit missing-core panic. Run it locally with:

```sh
cd /workspace/repo
./frontier-build.sh --output /workspace/repo/.frontier-build --harness /workspace/repo/frontier-example.rs
/workspace/repo/.frontier-build/target/debug/examples/frontier_probe
```

The adapter's first stdout line is the following actual output record; `shape` is the complete tensor shape, `access_axis` is its outer/item split, and `values` lists the row-major tensor elements:

```text
original shape=[1, 1, 2, 2], access_axis=4, values=[-4.0, -4.0, -4.0, -4.0]
```

It then runs your hooks and the mapper and prints `local Runner: iterations=..., nodes_before=..., nodes_after=...`. Those numbers depend on the submitted rules. This is a public input/debugging example, not a graded discovery test, and it contains no transformed target expression. Its small shapes and explicit numerical values are separate from the generated suites and their seeded assignments.

### Harness request and result records
The judge's probe reads one UTF-8 JSON request from standard input and writes one JSON result array to standard output. You implement the Rust constructors, not this parser or printer. A complete measurement request for the public example's shapes is:

```json
{
  "mode": "measure",
  "workloads": [
    {
      "id": "public-example",
      "data_shape": [1, 1, 3, 3],
      "weight_shape": [1, 1, 2, 2],
      "strides": [1, 1],
      "padding": [0, 0, 0, 0],
      "layout": "NCHW",
      "wrapper": 0,
      "seed": 0,
      "data_name": "data",
      "weight_name": "weights"
    }
  ]
}
```

| Workload field | Type and meaning |
| --- | --- |
| `id` | String identifying the result row; not a tensor symbol |
| `data_shape`, `weight_shape` | Four positive integer dimensions for convolution records; explicit records carry the first tensor shapes for compatibility |
| `strides` | Two positive integers `[Sh,Sw]` for convolution records; empty for explicit records |
| `padding` | Four nonnegative integers `[top,left,bottom,right]` for convolution records; empty for explicit records |
| `layout` | `"NCHW"` or `"NHWC"` determines convolution weight layout; `"EXPLICIT_OUTER_CONCAT"` identifies explicit records |
| `wrapper` | Integer selecting the surrounding operations below |
| `seed` | Unsigned 64-bit seed used for numerical verification; replaced with 0 in measurement requests |
| `data_name`, `weight_name` | Symbol strings used in `access-tensor` and as the keys of `name_to_shape` |
| `structure` | Optional explicit outer-concat record with four tensor specs, one concatenation axis, rank-two operand outer and item shapes, and no surrounding wrapper |

For explicit records, `structure.left` and `structure.right` each contain two tensor specs with fields `name`, `shape`, and `access_axis=2`. Corresponding left/right entries form each Cartesian product. The four-dimensional operand shapes divide into two outer and two item dimensions, so each product has four outer dimensions and item shape `[2]` followed by the common two-dimensional item shape. `structure.concat_axis` is 0 or 3. The judge builds `compute dot-product` directly over their concatenation. All four tensor names receive assignments from the same unchanged two-sample numerical protocol. No rule activity or number of mapped contractions beyond the existing minimum of one is required.

The wrapper is applied after producing the convolution output in its requested layout:

| `wrapper` | Original expression's surrounding operations |
| --- | --- |
| 0 | None |
| 1 | Transpose full-tensor axes by `[0,3,1,2]` |
| 2 | Insert a size-one axis at 0, then squeeze axis 0 |
| 3 | Transpose by `[2,0,3,1]`, then by `[1,3,0,2]` |

The measurement result has one object per workload, in request order. Its fields are:

| Result field | Type and meaning |
| --- | --- |
| `workload` | The input `id` string |
| `rewrite_time_seconds` | Positive, finite JSON number: the median of the three Runner times |
| `egraph_node_growth` | Nonnegative integer: maximum final-minus-initial node growth across three executions |
| `witnesses` | Array of exactly three entries, each a serialized Glenside S-expression string or `null`, one per repetition |

A separate verification request uses `"mode":"verify"`, repeats each workload three times with its numerical seed restored, and adds a top-level `witnesses` array aligned with those records. Its result is an equally long JSON array of booleans. The evaluator sets `im2col_discovery` to 1 only if all three booleans for that workload are true; otherwise it is 0. Witness strings and per-workload result rows are internal judge data, not the feedback returned to you. The public example's manually specified arrays are not conveyed by its measurement request.

### Build and measurement protocol
The judge builds a private probe against the retained `src/lib.rs` using:

```sh
/toolchain/bin/cargo build --manifest-path /build/code/Cargo.toml --locked --offline --no-default-features --lib --example frontier_probe
```

It sets `CARGO_BUILD_JOBS=1`, `CARGO_PROFILE_DEV_OPT_LEVEL=1`, `CARGO_PROFILE_DEV_DEBUG=0`, and `CARGO_NET_OFFLINE=true`, with isolated Cargo home and target directories. The resulting executable is invoked as `/probe`, with no command-line flags, working directory `/`, and the JSON request on stdin. Each measured process receives one workload and performs three searches. Numerical verification is a separate process.

Each repetition creates a fresh e-graph, adds the original expression, rebuilds it, and runs a fresh `egg::Runner` with the four rules in the specified order. The limits are 30 iterations, 100000 nodes, and 10 seconds per search. These are challenge constants (not from materials); hitting a Runner limit ends the search and still permits extraction, rather than itself invalidating the workload. External process and aggregate deadlines remain hard limits.

Timing is `Runner::report().total_time`, excluding compilation, process startup, extraction, and numerical validation. The workload time is the median of its three measurements. Growth is `Runner::report().egraph_nodes` minus `EGraph::total_number_of_nodes()` immediately after rebuilding the input. The reported growth is the maximum across the three executions, a conservative challenge-defined aggregation (not from materials). Growth or witness availability may vary without aborting other workloads; all three witnesses must still pass trusted verification for discovery credit. The input expression must remain in the original root e-class. Reported time must be finite and positive, and growth must be nonnegative.

For local compilation use `./frontier-build.sh --output /workspace/repo/.frontier-build`; `./frontier-bench.sh` additionally runs the remaining CPU language tests, not the private analysis-result coverage. Use the worked input/output example above and `frontier-example.rs` as the limited public interface examples. These helpers use two Cargo build jobs and apply the supplied CPU-only feature-gating repair only in a disposable build tree. Local test output is not the graded discovery measurement.

### Credited output and numerical correctness
The fixed mapper recognizes a dot product over a Cartesian product of shape-compatible accesses. Both operands must have one-dimensional item shapes; the first operand must have zero or one outer dimensions, and the second must have exactly one outer dimension. The mapper, not submitted code, introduces the `systolic-array` call.

The fixed extractor selects a minimum-cost complete expression from the original root e-class. Its additive cost is 1000000 for each `compute` node and 1 for every other node, plus its children's costs, using saturating integer addition. Credit concerns this selected expression, not any arbitrary node elsewhere in the graph or every possible extraction. A witness must contain at least one `systolic-array` or `systolic-array-with-blocking` node and fewer than 4096 expression nodes. Its serialization must be shorter than 262144 bytes. These are challenge constants (not from materials).

The entire witness must have exactly the original root's outer shape and item shape, including the surrounding wrapper's output layout. For numerical checking, the judge replaces mapped matrix calls with their tensor-operation meaning and executes that expression and the original on each of the two seeded tensor assignments. The resulting tensor dimensions, `access_axis`, and every `f64` element must agree exactly, with no numerical tolerance. Every intermediate access must agree with shape analysis and contain at most 2000000 tensor elements. Each repetition's witness must pass both assignments.

The numerical wrapper delegates tensor operations to the retained interpreter and materializes operands in row-major layout. It supplies shape-query support and makes `access-tensor` and `access-flatten` metadata follow `MyAnalysis`. In particular, the bare retained interpreter initializes `access-tensor` with `access_axis=0`, whereas the graded wrapper uses the analysis convention of all tensor dimensions being outer. These are different interfaces; the graded shape convention is the one specified above. The checker validates the selected complete witnesses, not every equality in the saturation closure. Submitted rewrites must nevertheless be semantics-preserving.

## Validity
Validation is applied before performance credit:

1. The artifact must satisfy the patch policy in `/frontier/task/README.md` and `/frontier/task/bin/submission-policy.json` and apply to a pristine tree. Changes outside the editable surface, altered or shadowed fixed mapping code, changes to its shape constraint, non-Rust extension files, and symbolic links invalidate the whole submission.
2. Candidate analysis and generic rewrites must be pure. After recognizing Rust comments, the source check rejects I/O, process, clock, external-source and symbol-redirection constructs: `unsafe`, `extern`, `include`, `include_str`, `include_bytes`, `env`, `option_env`, `asm`, `global_asm`, `print`, `println`, `eprint`, `eprintln`, `dbg`, `fs`, `io`, `net`, `process`, `thread`, `libc`, `Command`, `File`, `OpenOptions`, `SystemTime`, `Instant`, `link`, `link_section`, `no_mangle`, and `export_name`. These are conservative source checks, not permission to evade purity through another spelling. Analysis helpers may inspect every retained language variant, including both systolic-array forms and Relay/accelerator tags. Generic rewrite code may not construct accelerator or opaque-call nodes; the supplied ordinary and blocked mapping functions and their shape constraint must remain exact. At runtime, each submitted generic rule is checked for introduction of new accelerator or opaque-call nodes; only the separately registered fixed mappers may introduce mapped calls. No mandatory source decomposition is imposed.
3. Build failure, measurement crash, malformed measurements, an external timeout, or a resource/isolation violation scores zero for the entire evaluation, before performance is considered. Execution is isolated from judge files, credentials, and network access.
4. A missing witness or a witness that fails the bounded shape/numerical verification gives `im2col_discovery=0` for that workload. It does not by itself invalidate otherwise successful workloads. Interpreter failures caught during witness verification also mean no discovery credit for that workload.

## Scoring
Maximize the continuous score. For each workload `w`, let `D[w]` be its valid discovery indicator (0 or 1), `t[w]` its three-run median rewrite time in seconds, and `g[w]` its maximum node growth across those repetitions. For a valid evaluation on workload set `W`, the exact rule is:

```text
T[w] = (10.0 / t[w]) ** 0.125
G[w] = (100001.0 / (g[w] + 1)) ** 0.125
timing[w] = T[w] / (1 + T[w])
growth[w] = G[w] / (1 + G[w])
score = sum(D[w] * (60 + 20 * timing[w] + 20 * growth[w]) for w in W) / len(W)
```

Every workload has equal weight, including undiscovered workloads, whose contribution is zero. The constants `10.0`, `100001.0`, `0.125`, `1`, `60`, and `20` are challenge constants (not from materials). The time and node anchors come from the public search budgets; they are normalization scales, not measured comparator performance or guarantees about observed bounds. There is no additional threshold or private-comparator term in this formula. The score lies on a 0–100 scale, with a supremum of approximately 96.17 for nonnegative growth as time approaches zero; 100 is not attainable. Hard validity failures receive zero as specified above.

## Calibration
The following measurements are historical and precede this access-analysis-core successor. They are not current-source validation, a solver trial, or evidence of successor difficulty.

The current missing-core starter panics on analysis and is a hard evaluation failure: score 0 with a FAIL verdict, not fourteen measured no-discovery rows. The evaluator reason is `candidate analysis contract failed`; the submission interface exposes its approved feedback, not this internal message. A valid evaluation with no discoveries receives zero regardless of its recorded time or growth; undiscovered cases remain in the mean. As a formula check, a workload with `D=1`, `t=10` seconds, and `g=100000` contributes exactly 80 points, so a suite with those measurements on every workload scores 80. This is an arithmetic example using challenge constants (not from materials), not a measured target. Valid partial discovery receives the mean of its per-workload contributions; failing the separate stricter criterion does not replace that score.

Measured development calibration of the preceding authoring revision (not from materials): two **non-reference candidate patches**, A and B, were each evaluated three complete times using the same score formula on the authoring host restricted to two CPUs. That revision required equal repeated node counts; the current maximum-growth convention agrees when those counts are equal. These historical timings are not measurements of the current native image. Their evaluation order alternated; each workload's timing still used the prescribed median of three searches. Both patches discovered 12/12 cases in every evaluation, and each patch's per-workload node counts repeated exactly.

| Fixed candidate | Observed score range | Peak-to-peak rerun span |
| --- | --- | --- |
| A | 90.6798–90.6948 | 0.0150 points |
| B | 90.8534–90.8737 | 0.0203 points |

B's mean score was 0.1750 points higher, and its lowest score exceeded A's highest by 0.1586 points at unchanged discovery coverage. In this sample a same-patch change of roughly 0.01 point was consistent with rerun variation, whereas the separation between the two patches persisted across repeats. Node counts are exact integers; timings depend on the host. These are observed ranges from six development calls, not confidence bounds, stopping thresholds, or hardware-independent guarantees. They introduce **no tolerance**: the exact per-workload comparisons below remain unchanged.

## Feedback and grading
An accepted train submission returns a receipt immediately; read its completed development score and approved metrics with `frontier-submit status <verification_id>`. A saved answer is not evaluated by `store-final`, and the final score is not returned to the solver. At most 100 development submissions are allowed; the packaging and quota interface is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its seeds are disclosed. The two suites are stratified so they carry the same distribution, and they share no item.

The candidate measurements exposed in feedback are:

| Metric | Meaning |
| --- | --- |
| `development_im2col_discovered_count` | Number of the 14 feedback workloads with valid discovery in all three repetitions |
| `development_rewrite_time_seconds_median` | Median across workloads of your per-workload three-run median rewrite times, in seconds |
| `development_egraph_node_growth_median` | Median across workloads of your per-workload node-growth measurements |

Besides the continuous score, a separate stricter criterion is evaluated independently on every held-out workload and must pass on every workload. It compares your `im2col_discovery` (higher is better), `rewrite_time_seconds` (lower is better), and `egraph_node_growth` (lower is better) against a private comparator under the fixed search limits. The comparisons are `>=` for discovery and `<=` for time and growth, with no tolerance. All three comparisons must pass on every workload. This criterion does not zero or replace the continuous score; a submission that does not clear it is still scored normally.

For each of these three comparisons, feedback also gives the number and percentage of feedback workloads that passed and failed. This is aggregate per-metric progress, not an identification of the workloads. Raw per-workload measurements, witnesses, private-comparator measurements, and the overall stricter-criterion outcome are withheld.

## Resources
The environment is CPU-only Rust 1.85.1 with pinned `egg` 0.7.1 and offline Cargo dependencies. Its budgets are challenge constants (not from materials):

| Resource or limit | Budget |
| --- | --- |
| vCPUs | 2 |
| RAM | 4096 MiB |
| Storage | 8192 MiB |
| Initial environment build allowance | 600 seconds |
| Aggregate evaluation wall time | 300 seconds, including builds, measurement, and verification |
| Each isolated Cargo build | At most 180 seconds, also bounded by the remaining aggregate time |
| Each measurement or verification process | At most 60 seconds, also bounded by the remaining aggregate time |
| Each Runner search | 30 iterations, 100000 nodes, 10 seconds |
| Address space per isolated process | 3072 MiB |
| Per-file size limit in isolated processes | 64 MiB |
| Captured stdout per isolated invocation | At most 2097152 bytes |
| Answering allowance | 18000 seconds |
| Total train verification allowance | 12000 seconds |
| Independent final evaluation allowance | 300 seconds |

These submission clocks describe this native task's `policy.json`. The 100 calls
are a ceiling, not a guarantee that every call can use its full evaluation limit.
The 12000-second train pool is shared by all calls and included in the
18000-second answering window; it does not extend that window. Platform train
time includes startup and transport as well as evaluation. Each evaluator
invocation keeps its 300-second aggregate limit, and a nearly exhausted train
pool or answering window can end a call earlier. Final evaluation has its own
300-second timeout after the Agent exits, independent of remaining answering time.

The aggregate evaluation limit takes precedence over the sum of individual allowances. Each process also has a CPU-time limit equal to its allotted wall-time cap rounded up to whole seconds. The measurement-process allowance covers all three repetitions and extraction; numerical verification has its own process cap. The Linux kernel must support Landlock, and isolation fails closed if unavailable.
