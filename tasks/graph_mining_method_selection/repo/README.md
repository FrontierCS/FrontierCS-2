# Graph Mining Method Selection

Supply ScaleGPM's engine selector: for each approximate graph-pattern-mining configuration,
return `"GS"` or `"NS"` so that the selected engine's stored runtime is as small as possible.
Both engines meet the error target in every case, so either answer is correct; the score rewards
low geometric-mean runtime regret relative to the better constant selector.

## Task

ScaleGPM (Chen et al., "Accurate and Fast Approximate Graph Pattern Mining at Scale", PVLDB)
estimates pattern counts within a relative-error bound with two CPU/OpenMP engines: neighborhood
sampling (`NS`) and graph sparsification (`GS`, color sparsification then exact counting). Its
hybrid stage chooses the engine per workload. The `threshold` selector in
`performance_model/src/main.py` has been removed.

## What you may change

- Only `performance_model/src/main.py`: the `threshold` callable and any supporting Python
  (functions, classes, constants, standard-library imports) in that file.
- No file may be added or deleted; the evaluator protocol, `data.py` and both mining engines
  stay unchanged. No third-party Python package and no external service is available to the
  selector.

## Interface

```python
def threshold(graph: str, pattern: str, error: float,
              ns: int | None = None, c: float | None = None,
              cores: int = 1) -> str
```

For each case the judge sets `data.graphs["workload"] = data.Graph("measured", v, e, triangles,
max_degree, 1.0)` and calls `threshold("workload", pattern, error, ns, c, cores)` with all six
arguments positionally. It must return exactly `"GS"` or `"NS"`.

| Argument | Meaning |
| --- | --- |
| `graph` | key in `data.graphs`; always `"workload"` |
| `pattern` | key in `data.patterns`: `"triangle"` or `"4clique"` |
| `error` | requested relative error: `0.1` or `0.2` |
| `ns` | NS sample budget |
| `c` | GS number of colors |
| `cores` | CPU parallelism of the measured configuration |

The `Graph` exposes `name` (`"measured"`), `v` (vertices), `e` (symmetric adjacency entries,
twice the undirected edges), `tri_cnt` (exact triangle count), `max_degree` and
`mem_look_up_time` (`1.0`). The first train case is `v=60000, e=839976, triangles=900082,
max_degree=20, pattern="triangle", error=0.2, c=4, ns=41447, cores=1`, i.e.
`threshold("workload", "triangle", 0.2, 41447, 4, 1)`.

The judge runs `python3 -I -c <runner> <tree>`, imports `data` and `main` from
`<tree>/performance_model/src`, feeds the cases on standard input and calls `threshold` in table
order; no graph identifier, suite label, index, stored runtime or stored error is passed. The
selector may also read the `data.patterns` entries. What it prints is ignored, as long as the
runner can still write its final result line.

## Correctness

The evaluation scores 0 if `main.py` fails to import, `threshold` raises, any call returns
anything but `"GS"` or `"NS"`, the process crashes, or import plus all 112 calls do not finish
within one 12-second deadline (Python 3.12, standard library only). There is no partial score.

## Workload

A fixed table of 224 measured configurations: 14 per graph for 16 deterministic social-like
graphs (50,000–180,000 vertices, 779,980–5,939,690 adjacency entries, 151,473–15,016,682
triangles, maximum degree 19–54). Each configuration has `pattern` triangle or 4clique, `error`
0.1 or 0.2, `ns` log-uniform in about 7,000–500,000, `c` an integer 1–20 and `cores` 1, 2, 4 or
8. Train and final each use 112 cases from eight graphs, with no graph or configuration shared.
Stored runtimes are medians of three runs of the authors' OpenMP backends; nothing is timed live.

## Scoring

For the `n = 112` cases with stored runtimes `t_i(GS)`, `t_i(NS)` and choice `s_i`:

```text
r_i       = t_i(s_i) / min(t_i(GS), t_i(NS))
R         = exp((1/n) * sum_i log(r_i))
B         = min(R of always-"GS", R of always-"NS")   on the same suite
unbounded = 100 * (B - R) / (B - 1)
score     = min(100, max(0, unbounded))
```

Always choosing the faster engine scores 100; matching the better constant scores 0 (on the
train suite always-`"GS"` gives `B = 1.254618`). The score is deterministic for a fixed suite.

**Beats the reference** when, separately for the triangle and the 4-clique cases, it has `selected_runtime_seconds`: a geometric mean
selected runtime strictly below the reference selector's (only a 1e-12 relative guard; ties
fail), and `accuracy_qualified`: every selected estimate within its error target.

## Running locally

```sh
cd /workspace/repo/performance_model/src && python3 -c 'import data, main; \
data.graphs["workload"] = data.Graph("measured", 60000, 839976, 900082, 20, 1.0); \
print(main.threshold("workload", "triangle", 0.2, 41447, 4, 1))'
```

## Notes

The cases are graph profiles, not inputs to recognize: do not special-case the listed train
configurations.
