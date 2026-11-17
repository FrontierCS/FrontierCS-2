# Changes from the original sources

This tree is derived from an MIT-licensed open-source project (see LICENSE).
For this task:

- Method-specific code, configurations, precomputed budgets, documentation and
  examples were removed. Only the model and decoding code (`models/`), generic
  helpers (`evaluation/utils.py`), and the QuALITY loader
  (`evaluation/datasets.py`) remain, together with the public development data.
- `compaction/` is new: the solver interface and a random-eviction starting
  point.
- `frontier-bench.sh` and `frontier_bench.py` are new: a public development
  check.
