# Join-Tree Rewriting

Implement `rewrite_plan`, which turns an optimizer's binary equijoin tree into a tree that lowers to a two-phase nested-semijoin (Yannakakis) plan, and make the lowered plans run fast. A valid rewrite keeps every leaf and join predicate, satisfies the lowerability invariant and returns exactly the original query result.

## Task

The system is the artifact of "Instance-Optimal Acyclic Join Processing Without Regret" (PVLDB 2025): a Python planner converts each query's topmost binary join subtree into `BinaryJoinNode`/`LeafNode` objects, your `rewrite_plan` returns a rewritten tree, and a fixed pipeline lowers it to a nested-semijoin physical plan executed by a single-threaded columnar engine (`join_runner`). The shipped `rewrite_plan` is an identity stub; the original rewrite policy was removed and is the reference.

## What you may change

- `2phase_nsa/binary_plan/rewrite_policy.py` only: replace the body of `rewrite_plan` and add private functions, classes, type aliases and constants. Do not delete the file or change its mode.
- The module must parse as Python, define `rewrite_plan`, and import only from `typing` and the shipped `binary_plan` package.
- It calls none of `open`, `exec`, `eval`, `compile`, `input`, `breakpoint`, `__import__`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, and accesses no attribute whose name begins with `__`.
- Its source text does not contain `/proc`, `/sys`, `frontier_submission`, `plan_builder` or `evaluator.py` (any letter case).
- A violation of any of the three rules above is rejected as `content_policy`; deleting or editing another file is `out_of_scope`, and a mode change is `forbidden_file_type`.
- Fixed: the node classes, parsing, equivalence classes, scans, filters, projections, lowering, the engine, the data and the timing. Standard output and error are discarded.

## Interface

`2phase_nsa/binary_plan/rewrite_policy.py`:

```python
def rewrite_plan(plan: BinaryJoinNode | LeafNode) -> BinaryJoinNode | LeafNode
```

It is called once per query and returns the tree in memory; it writes no file. `LeafNode(relation_name: str, attributes: set[int], ec: int)` is a scan: its alias, the equivalence-class IDs in its projected schema, and its estimated cardinality. `BinaryJoinNode(left_child, right_child, join_attributes: set[int])` is a join whose output schema is the union of its children's attributes. Returned objects need not be the input objects.

Each query is a JSON plan with `execution_time`, `root` and `aliases`; `binary_plan.util.ir_to_binary_plan` follows `root` to the topmost `HASHJOIN`, builds equivalence classes from every join `condition`, and converts the subtree. A scan record has `relation`, `projection` (fields with query-local aliases), `opt_filter` (or `null`), `estimated_cardinality` and empty `children`; a join has two `children` and `condition` pairs equating one field from each side:

```json
{
  "name": "HASHJOIN",
  "execution_time": 0.002087,
  "actual_rows": 41569,
  "estimated_cardinality": null,
  "children": [
    {
      "name": "SEQUENTIALSCAN",
      "execution_time": 0.000058,
      "actual_rows": 40324,
      "estimated_cardinality": 40325,
      "children": [],
      "relation": "users",
      "opt_filter": "Id<=55746 AND Id IS NOT NULL",
      "projection": [
        {"table_name": "u", "field_name": "Id"}
      ]
    },
    {
      "name": "SEQUENTIALSCAN",
      "execution_time": 0.000751,
      "actual_rows": 42291,
      "estimated_cardinality": 18396,
      "children": [],
      "relation": "posts",
      "opt_filter": "PostTypeId=1 AND PostTypeId IS NOT NULL AND Score>=-1 AND Score IS NOT NULL AND CommentCount>=0 AND CommentCount<=11 AND CommentCount IS NOT NULL",
      "projection": [
        {"table_name": "p", "field_name": "Id"},
        {"table_name": "p", "field_name": "OwnerUserId"}
      ]
    }
  ],
  "condition": [
    [
      {"table_name": "u", "field_name": "Id"},
      {"table_name": "p", "field_name": "OwnerUserId"}
    ]
  ]
}
```

Example: `U = LeafNode("u", {1}, 40325)`, `P = LeafNode("p", {0, 1}, 18396)`; for `plan = BinaryJoinNode(U, P, {1})`, returning `BinaryJoinNode(U, P, {1})` is valid.

## Correctness

The returned tree must:
- contain each original alias exactly once, with the same attribute set and estimated cardinality, and exactly `number_of_leaves - 1` joins;
- give each join the intersection of the attributes below its two children;
- satisfy, for every `BinaryJoinNode(P1, P2, J)`, `J ⊆ LA(P1)` and `J ⊆ LA(P2)`, where `LA(Pi)` is the attribute set of the left-most leaf below `Pi` (`binary_plan.binary_plan.is_well_behaved`);
- be accepted by the fixed lowering and return exactly the original binary plan's result.

A query whose rewrite raises, breaks these rules, exceeds 30 s of plan construction (each of its two conversion subprocesses 20 s) or returns a different result scores 0 and fails both PASS comparisons. It stays in the denominator, and a leftover output file is not accepted. An empty patch evaluates the identity stub, which fails lowering on every query.

## Workload

One workload is one query plan from three benchmarks over 95 Parquet relations held by the judge (not in the Agent image). Every plan is an acyclic equijoin of 5 to 14 uniquely aliased leaves with no Cartesian products or non-equality joins, and initially violates the invariant.

| benchmark | train (public) plans | added in final (hidden) | leaves |
|---|---:|---:|---:|
| Stats-CEB | 6 | 6 | 5–6 |
| Join Order Benchmark (IMDB) | 5 | 4 | 9–14 |
| Cardinality-estimation graph benchmark | 9 | 9 | 6–8 |

Train runs the 20 public plans; final freshly runs those 20 plus the 19 hidden ones (39). Public and hidden plans are stratified by join size and share no query.

Per query, one untimed correctness pass runs binary, reference and candidate. The timing pass then repeats BinaryA, ReferenceA, CandidateA, CandidateB, ReferenceB, BinaryB five times, giving ten observations per role. Each observation times plan execution and result collection, not plan construction, loading or formatting, and results are re-compared every time.

## Scoring

```
B_q = median(binary durations)      C_q = median(candidate durations)
R_q = median(reference durations)   S_q = B_q / C_q
L(S_q) = 0 if S_q <= 1, else 100 * log(S_q) / (log(10) + log(S_q))
score = sum(L(S_q)) / N             N = 20 (train) or 39 (final)
```

A failed, unmeasured or incorrect query contributes 0. The score is 0 at or below the binary baseline (the original plan run afresh) and 50 at a tenfold speedup on every query. It is unclipped and unrounded.

**Beats the reference** requires two conditions on every query:
- `exact_result`: the candidate returns the original result in every run;
- `median_latency`: `C_q < R_q` strictly, with no tolerance, against the reference measured in the same evaluation. A plan byte-identical to the reference's fails.

## Running locally

```sh
cd /workspace/repo/2phase_nsa
python3 -c 'from binary_plan.binary_plan import *; from binary_plan.rewrite_policy import rewrite_plan; U=LeafNode("u",{1},40325); P=LeafNode("p",{0,1},18396); t=rewrite_plan(BinaryJoinNode(U,P,{1})); print(pretty_print(t), is_well_behaved(t))'
```

The engine and the data are judge-only: latency and result correctness can only be measured by a train submission.

## Environment

Python 3.12 for both the plan builder and your module. The engine is the artifact's reproduced Rust `join_runner`, and each cumulative engine phase has a 600 s limit (an engine timeout is an evaluation error, not a candidate score).
