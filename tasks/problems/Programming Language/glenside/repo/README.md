# Access-Pattern Rewriting

Implement Glenside's compositional access-pattern analysis (`MyAnalysis::make` and
`MyAnalysis::merge`, now stubs) and, optionally, improve the four generic rewrites, so that equality
saturation maps small convolution and concatenation workloads onto the fixed systolic-array mapper.
**Maximize the mean score**: valid accelerator discovery first, then lower rewrite time and node
growth. A submission must compile and pass the analysis-contract check; only numerically matching
witnesses count.

## Task

Glenside (Smith et al., "Pure Tensor Program Rewriting via Access Patterns", MAPS 2021) is a Rust
tensor IR built on the `egg` e-graph library. An access pattern splits a tensor into an outer shape,
which indexes items, and an item shape, the array at each position. The analysis attaches a typed
fact to each e-class, which rewrites and the mapper read. The fixed mapper turns a dot product over
a Cartesian product of suitably shaped accesses into a `systolic-array` call. No accelerator is
executed; the task is CPU-only.

## What you may change

- `code/src/language/solver_extensions/access_analysis.rs`: the `egg::Analysis` implementation.
- `code/src/language/rewrites.rs`: the four generic rewrite constructors and helpers.
- New `.rs` files under `code/src/language/solver_extensions/`.

Everything else is fixed. In `rewrites.rs`, `systolic_array` and `systolic_array_with_blocking`
stay byte-identical and defined once, and `constrain_access` verbatim (else `out_of_scope`).
Rejected as `content_policy`:
- in `rewrites.rs` outside those two functions, comments included: `SystolicArray` and any word
  starting with it, `RelayOperatorCall`, `AcceleratorCall`;
- there and in extension files, outside comments (strings count), the whole words `unsafe`,
  `extern`, `asm`, `global_asm`, `print`, `println`, `io`, `link`, `link_section`, `no_mangle`,
  `export_name`;
- a `path = "..."` other than a relative `solver_extensions/….rs` path without `..`.

## Interface

```rust
impl egg::Analysis<Language> for MyAnalysis {
    type Data = MyAnalysisData;
    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge;
}
```

`make` builds an enode's fact from its children's facts and `name_to_shape` / `name_to_dtype`
(the latter empty; dtype defaults to float32). Access facts hold outer `shape`, `item_shape`,
`zero_regions`, `access_pattern_shape_settled` and `contains_accelerator_calls`. Match pinned
upstream behaviour, not its TODOs:
- `merge` updates `to`, returns `DidMerge` flags per input, honours settledness, propagates the
  accelerator flag, merges tuples recursively, and is stable under repeated merges and rebuilds.
  Access merges need equal element counts and, if both are settled, equal shapes; violations
  panic. Keep the target's zero regions (no union).
- Non-reshape access nodes are settled iff all access children are; a reshape may stay unsettled
  but never settles over unsettled children.
- Systolic-array branches pass on their children's accelerator flag; the node does not set it.
- Zero regions must be sound, not complete.
- Dot product needs an item dimension and keeps the outer shape with item `[]`, whatever the
  child: `(compute dot-product (access (access-tensor x) 1))`, `x: [2,3,4]`, gives `[2]`/`[]`, not
  a panic. Reshape does not check element counts.

Facts, merge results and merge flags are compared with the reference's on private cases, except
merge flags for two unsettled accesses or zero-region-only differences, and per-node settledness
and zero regions beyond the rules above. Malformed IR outside the workload and rewrite domain need not be
rejected.

Rewrites (`RW = egg::Rewrite<Language, MyAnalysis>`), registered in this order before the judge's
own `systolic_array()` and `systolic_array_with_blocking(1, 1)`:

```rust
pub fn flatten_unflatten_any_access() -> RW
pub fn bubble_reshape_through_cartesian_product() -> RW
pub fn bubble_reshape_through_compute_dot_product() -> RW
pub fn bubble_access_concatenate_through_compute_dot_product_not_item_axis() -> RW
```

Each must be callable when inapplicable and never panic, equate unequal values or mispair
Cartesian-product items. No rule has to fire. The mapper matches a dot product over a Cartesian
product of 1-D-item operands, the first with at most one outer dimension, the second with exactly
one.

Operators (full shape `outer ++ item`; `access_axis` = outer rank; axes zero-based; item `[]` =
scalar):

| Operator | Result |
| --- | --- |
| `(access a k)` | split after `k` full dimensions; `(access-tensor n)` has all dimensions outer |
| `(access-windows a (shape K…) (shape S…))` | outer `A`, item `I` → outer `A ++ Q`, item `K`, `Q[j] = floor((I[j]-K[j])/S[j])+1`, `0 < K[j] <= I[j]`, `S[j] > 0` |
| `(access-cartesian-product l r)` | equal items `I` → outer `A ++ B`, item `[2] ++ I` |
| `(compute dot-product a)` | items `[m] ++ I` → product of the `m` values, summed over `I` |
| `access-squeeze`, `access-insert-axis` | remove / insert a size-1 full axis |
| `access-transpose a (list p…)` | permutes full axes, keeps `access_axis` |

Worked example: NCHW data `[1,1,3,3]` = `[1..9]`, OIHW weights `[1,1,2,2]` = `[1,0,0,-1]`, unit
stride, no padding, give `[1,1,2,2]` = `[-4,-4,-4,-4]`. The original expression is:

```lisp
(access-transpose
  (compute dot-product
    (access-cartesian-product
      (access (access-tensor weights) 1)
      (access
        (access-squeeze
          (access-windows
            (access (access-pad (access-pad (access-tensor data) zero-padding 2 0 0)
                                zero-padding 3 0 0) 1)
            (shape 1 2 2) (shape 1 1 1))
          1)
        3)))
  (list 1 0 2 3))
```

| Subexpression | Outer | Item |
| --- | --- | --- |
| data after `access … 1` | `[1]` | `[1,3,3]` |
| windows | `[1,1,2,2]` | `[1,2,2]` |
| after squeeze and `access … 3` | `[1,2,2]` | `[1,2,2]` |
| weights after `access … 1` | `[1]` | `[1,2,2]` |
| Cartesian product | `[1,1,2,2]` | `[2,1,2,2]` |
| dot product, transpose | `[1,1,2,2]` | `[]` |

## Correctness

Scored 0 for the whole evaluation: a compile failure with every error in your editable files, or a
failed analysis contract (a fact differs from the reference, a contract assertion fails, or
analysis panics in your files, as the starter's `unimplemented!` stubs do).

No score: any other build failure (e.g. a changed signature breaking the judge's probe) or a build
over 180 s; an analysis crash elsewhere; a measurement crash or one over 60 s; a non-finite or
non-positive time; a process over 3072 MiB of memory, 64 MiB per file or 2097152 bytes of
stdout; the evaluation timeout. A measurement crashes if a generic rule adds an accelerator or
opaque-call node (`SystolicArray…`, `AcceleratorCall`, `RelayOperatorCall`) not already present,
or the original leaves the root e-class. A witness that fails verification only sets `D = 0` for
that workload (see Scoring).

## Workload

A workload is one expression over symbolic tensors of known shape. Train and final each have 14
from one generator with different seeds (same distribution, no shared item; records and seeds of
both suites withheld); 12 are convolutions from the retained `conv2d` constructor:

| Quantity | Values |
| --- | --- |
| `N`; `C`; `O` | 1, 2, 3; 2, 3, 4, 5; 2, 3, 5, 6 |
| `H`, `W` | each 6–10 |
| kernel / stride | `(1,1)/(1,1)`, `(2,3)/(2,1)`, `(3,2)/(1,2)`, `(2,2)/(2,2)` |
| padding `[top,left,bottom,right]` | 0 for `(1,1)` kernels, else each 0–2 |
| layout | 6 NCHW data `[N,C,H,W]` with OIHW weights; 6 NHWC `[N,H,W,C]` with HWIO |
| wrapper (3 each) | 0 none; 1 transpose `[0,3,1,2]`; 2 insert axis 0 then squeeze 0; 3 transpose `[2,0,3,1]` then `[1,3,0,2]` |

Dilation is 1, groups 1, no bias. The output before the wrapper is `[N,O,Ho,Wo]` or `[N,Ho,Wo,O]`
with `Ho = floor((H+top+bottom-Kh)/Sh)+1` and `Wo = floor((W+left+right-Kw)/Sw)+1`.

The other 2 are `(compute dot-product (access-concatenate P0 P1 axis))`, where each `Pi` is a
Cartesian product of two `(access (access-tensor t) 2)` with rank-2 outer and item shapes, extents
in `{2,3}`. Items and non-concatenated outer dimensions match; one concatenates on axis 0 (first
left-outer), one on axis 3 (last right-outer).

Each workload runs three searches on a fresh e-graph and `egg::Runner` limited to 30 iterations,
100000 nodes and 10 s (a limit just ends the search). Time is
`Runner::report().total_time`, median of three; nothing else is timed. Growth is final minus
initial node count, maximum of three. The reference is measured alternately in the same evaluation.

## Scoring

For workload `w` with discovery `D` (0/1), median time `t` seconds and maximum growth `g`:

```text
T = (10.0 / t) ** 0.125
G = (100001.0 / (g + 1)) ** 0.125
score_w = D * (60 + 20 * T / (1 + T) + 20 * G / (1 + G))
score   = mean of score_w over the 14 workloads of the suite
```

`D = 1` needs, in all three repetitions, a root extraction (1000000 per `compute` node, 1 per
other, saturating) with a `systolic-array` or `systolic-array-with-blocking` node, under 4096
nodes and 262144 serialized bytes, the original's outer and item shapes, and, unmapped, exact
equality (shape, `access_axis`, every `f64`) on two seeded assignments with values in
`{-3/8,…,3/8}`. Every intermediate access must match reference analysis and hold ≤ 2000000
elements.

Undiscovered workloads count as 0. Anchors: the starter scores 0; `D=1, t=10, g=100000` gives 80;
the supremum is about 96.17 (`t → 0`, `g = 0`).

**Beats the reference** when, on every final workload, discovery `>=`, time `<=` and growth `<=`
the reference's, with no tolerance. This does not change the score.

## Running locally

```sh
cd /workspace/repo
./frontier-build.sh --output /workspace/glenside-build      # build the library
./frontier-build.sh --output /workspace/glenside-build --harness /workspace/repo/frontier-example.rs
/workspace/glenside-build/target/debug/examples/frontier_probe
./frontier-bench.sh                                               # CPU language tests
```

The example prints `original shape=[1, 1, 2, 2], access_axis=4, values=[-4.0, -4.0, -4.0, -4.0]`,
then runs your rules and the ordinary mapper and prints Runner iterations and node counts.

## Environment

CPU-only Rust 1.85.1, `egg` 0.7.1, offline Cargo dependencies. The judge builds with
`--locked --offline --no-default-features`, `CARGO_PROFILE_DEV_OPT_LEVEL=1`,
`CARGO_PROFILE_DEV_DEBUG=0`, one job; its processes have no network and a CPU-time limit equal to
their wall-time cap.

## Notes

No workload lookup tables, fabricated facts or dedicated convolution lowering.

The bare interpreter starts `access-tensor` with `access_axis = 0`; the judge's check uses the
analysis convention (all dimensions outer).
