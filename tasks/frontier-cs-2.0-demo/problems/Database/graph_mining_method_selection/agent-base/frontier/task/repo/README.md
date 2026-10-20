# Graph Mining Method Selection

## Problem
ScaleGPM is an approximate graph-pattern mining system. Graph-pattern mining counts occurrences of a small pattern, such as a triangle, inside a much larger graph; approximate mining estimates that count within a requested relative-error bound. ScaleGPM includes two CPU/OpenMP mining engines: neighborhood sampling (`NS`) draws local samples, and graph sparsification (`GS`) color-sparsifies the graph and counts the pattern in the resulting graph. The system's hybrid-selection stage chooses which engine processes a workload. This task reads graph profiles and mining parameters for 224 measured configurations derived from sixteen deterministic, reduced, social-like graphs generated for the challenge. Each evaluated profile specifies triangle or 4-clique counting, a 10% or 20% error target, an NS sample budget, a GS color parameter, and an available-core count. For every call, the selector must return exactly `"GS"` or `"NS"`; both stored engine results meet the requested error target, so a correct output is either of those two strings. The objective is to maximize the score by minimizing geometric-mean runtime regret over the 112 cases in the applicable suite. The `threshold` selector in `performance_model/src/main.py` has been removed and must be supplied.

## Design space
The judged surface is the `threshold` callable and any supporting Python definitions placed in `performance_model/src/main.py`. Candidate code may use all six call arguments and the documented attributes of the selected `Graph` and `Pattern` objects. It may add functions, classes, constants, standard-library imports, and other Python code within that file. No additional third-party Python package is supplied for evaluation. The evaluator imports `data` and `main` in an isolated interpreter and calls `threshold` directly; it does not invoke the application CLI or either C++ executable. The submitted program must be self-contained in the allowed patch, must not require an external service, and must leave the evaluator protocol and both mining engines unchanged.

## Workload
The fixed measured table has 224 records: fourteen independently randomized configurations for each of sixteen graph profiles. These are challenge-generated, reduced social-like graphs rather than the paper's full public graph corpus. A submission is evaluated on 112 records from eight profiles. The feedback and graded suites use disjoint profiles and independently drawn configuration tuples from the same distributions. The generated graphs are undirected, represented by symmetric adjacency entries, and contain no self-loops or duplicate edges. Across all 224 records, their profiles span:

- 50,000 to 180,000 vertices;
- 779,980 to 5,939,690 adjacency entries;
- 151,473 to 15,016,682 triangles; and
- maximum degree 19 to 54.

Cases use `pattern="triangle"` or `pattern="4clique"` and `error=0.1` or `error=0.2`. Across the table, `ns` was drawn independently from a log-uniform distribution spanning approximately 7,000 to 500,000 (the observed range is 7,119 to 484,631), `c` is an integer from 1 through 20, and `cores` is one of 1, 2, 4, or 8. Each stored runtime is the median of three executions of the corresponding authors' OpenMP backend. A `GS` runtime covers sparsification and exact search; an `NS` runtime covers edge-stream construction and sampling. Both estimates in every stored record were compared with an exact unsparsified count and have relative error at or below that record's target.

The workload represents method selection on graph profiles, not a set of input strings to recognize. The runner does not pass a graph identifier, suite label, record index, stored runtime, or stored error to `threshold`. It invokes the cases sequentially in their fixed table order. GPU and distributed execution, patterns other than triangle and 4-clique, error targets other than 10% and 20%, and live timing of candidate-selected engines are not part of the graded workload. The 224-record reduced workload, graph generator, fourteen randomized configurations per graph, three timing repetitions, and 112/112 suite split are challenge choices rather than measurements specified by the paper.

## Interface
Supply a callable that accepts these six positional arguments; retaining the shown names and defaults also preserves the package's public interface:

```python
def threshold(graph: str, pattern: str, error: float,
              ns: int | None = None, c: float | None = None,
              cores: int = 1) -> str
```

For each case, the harness installs a `data.Graph` under the key `"workload"` and calls:

```python
threshold("workload", pattern, error, ns, c, cores)
```

The arguments have these meanings:

| Argument | Meaning |
| --- | --- |
| `graph` | Key in `data.graphs`; the judge always uses `"workload"`. |
| `pattern` | Key in `data.patterns`; evaluated records use `"triangle"` or `"4clique"`. |
| `error` | Maximum requested relative error; evaluated records use `0.1` or `0.2`. |
| `ns` | Sample budget supplied to the NS engine. |
| `c` | Number-of-colors parameter supplied to the GS engine. |
| `cores` | CPU parallelism for the measured configuration. |

The selected `data.Graph` exposes `name`, `v`, `e`, `tri_cnt`, `max_degree`, and `mem_look_up_time`. During evaluation, `name` is `"measured"`, `tri_cnt` receives the record's `triangles` value, and `mem_look_up_time` is `1.0`.

Here is the first actual feedback record after the evaluator keeps only the nine public fields:

```json
{
  "v": 60000,
  "e": 839976,
  "triangles": 900082,
  "max_degree": 20,
  "pattern": "triangle",
  "error": 0.2,
  "c": 4,
  "ns": 41447,
  "cores": 1
}
```

`v` is the vertex count, `e` is the number of symmetric adjacency entries (twice the number of undirected edges), `triangles` is the exact triangle count recorded in the graph profile, and `max_degree` is the graph's maximum degree. The remaining fields map directly to the callable arguments documented above. For this record the harness performs the equivalent of:

```python
data.graphs["workload"] = data.Graph(
    "measured", 60000, 839976, 900082, 20, 1.0
)
answer = threshold("workload", "triangle", 0.2, 41447, 4, 1)
# answer must be exactly "GS" or "NS"
```

If that call returns `"NS"`, its matching collected output value is the JSON string `"NS"`. After all 112 calls, the runner writes one internal result record to standard output in this layout:

```text
FRONTIER_RESULT=["NS", "GS", ..., "NS"]
```

The list must contain exactly 112 valid strings. Candidate diagnostic output is ignored unless it prevents the harness from obtaining that final record.

The judge does not run the module's application CLI. It launches the active Python 3.11 executable as `<python> -I -c <judge-runner> <pristine-worktree>`, sends a JSON array containing the 112 public records to the runner on standard input, imports `data` and `main` from `<pristine-worktree>/performance_model/src`, and invokes `threshold` with all six arguments positionally as shown above. There are no additional command-line flags.

## Validity
The submitted patch must apply cleanly, modify only `performance_model/src/main.py`, be at most 65,536 bytes, change at most one file, contain no binary hunk, and use neither absolute nor parent-traversing paths. It must be a textual diff against the supplied repository state and must not be empty. Candidate code must import with the available standard-library Python environment, accept the evaluator's documented six-argument positional call, and return exactly `"GS"` or `"NS"` on every call. Import, input decoding, all 112 calls, result encoding, and process exit share one 12-second deadline.

Patch checks run before candidate import. A patch-policy violation, build or patch-application failure, syntax or import failure, exception, crash, timeout, missing result, malformed result, invalid decision, or resource-limit violation scores zero before performance is considered. There is no partial score after a hard validity failure.

## Scoring
The goal is to maximize the score on a 0-to-100 scale.

For each of the `n = 112` cases, let `t_i(GS)` and `t_i(NS)` be the stored, accuracy-qualified median runtimes, and let `s_i` be the submitted choice. Runtime regret is

```text
r_i = t_i(s_i) / min(t_i(GS), t_i(NS))
```

The aggregate geometric-mean regret is

```text
R = exp((1 / n) * sum(i=1..n, log(r_i)))
```

The scale is anchored on what costs nothing. Let `B` be the geometric-mean regret of the better of the two constant selectors — answer `"GS"` for every case, or `"NS"` for every case — computed on the same suite being graded. The evaluator computes, symbol for symbol,

```text
unbounded = 100.0 * (B - R) / (B - 1.0)
score     = min(100.0, max(0.0, unbounded))
```

Choosing the faster engine on every case gives `R = 1` and scores 100. Matching the better constant gives `R = B` and scores exactly 0. A selector that is worse than a constant scores 0, and `score_unbounded` reports how far below it fell. `B` is recomputed per suite because the two halves are not equally decidable — a constant wins 81 of 112 cases on one and 73 of 112 on the other — so a shared anchor would move scores between suites for reasons unrelated to the submission. The scale factor and the clamp are challenge constants (not from materials). The arithmetic uses the fixed paired-runtime table and is deterministic. Train feedback reports the score, candidate-selected absolute geometric-mean runtime in seconds, maximum selected relative error, case count, and per-metric `beat_spec_progress`. Regret ratios, oracle decision accuracy, reference measurements, and private workload identities are withheld.

## Beating the reference

Beating the reference requires strictly lower geometric-mean selected runtime separately on the triangle and 4-clique workloads, with every selected estimate meeting its original error target. All 112 configurations retain their existing pattern group and weight; both patterns must pass both metrics. Only a 1e-12 relative arithmetic guard is used and runtime ties fail. The continuous regret score is unchanged. Train feedback exposes absolute candidate summary measurements and per-metric workload pass/fail counts and percentages, without reference values, private identifiers, regret ratios or per-pattern decisions.

## Calibration
The two constant selectors set the scale on the feedback suite. Returning `"GS"` for all 112 cases chooses the faster stored engine in 81 cases and has geometric-mean regret approximately `1.254618`; returning `"NS"` chooses the faster engine in 31 cases with regret approximately `2.053708`. The better of the two is therefore `B ≈ 1.254618`, and it scores exactly `0`. There are no runtime ties in that suite.

The headroom on this task is narrow and stated rather than hidden: a perfect selector reaches `R = 1`, and the whole distance from the free constant to perfect is the `0`-to-`100` range. Decision accuracy is a private diagnostic, not a scoring term — cases enter the formula weighted by their runtime regret, so a wrong call on a case where the two engines are within noise costs almost nothing while a wrong call on a case with a wide gap costs a lot. The judge performs no live engine timing, sampling, or repeated candidate execution, so the score for unchanged code and a fixed suite has no run-to-run metric noise; a repeatable change of `0.1` score points is a genuine difference in the measured objective. No target score is disclosed.

## Feedback and grading
Train submissions return a score, the approved absolute candidate summaries, and `beat_spec_progress` for each of the two metrics. Each progress object reports `passed`, `failed`, `total`, `pass_percent`, and `fail_percent` over exactly two pattern workloads; all 112 configurations contribute to these workloads. Final submissions do not return Agent-visible metrics. The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its configuration draws are disclosed. Feedback and graded suites each contain 112 cases from eight graph profiles, with no graph profile or exact configuration tuple shared between them.

## Resources
The environment provides 8 vCPUs, 16,384 MB of memory, and 4,096 MB of storage. The build timeout is 300 seconds, the candidate process has one 12-second deadline, and the overall evaluation timeout is 900 seconds. The runtime policy is Python 3.11 on Ubuntu 24.04, with Python, Git, and the GNU build toolchain available; the supplied C++ engines use OpenMP.

Train returns these candidate-only absolute summaries: `geomean_selected_runtime_seconds`, `max_selected_relative_error`, `cases`. For each comparison metric, `beat_spec_progress` contains `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`; the denominator is the fixed workload count for the selected suite. No reference measurements, private workload identities, per-workload decisions, or margins are returned.
