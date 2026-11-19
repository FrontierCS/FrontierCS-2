# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it computes
the recalls and distance errors from the returned results. The train judge's image leaves out the
hidden workloads' datasets, sizes and seed.
