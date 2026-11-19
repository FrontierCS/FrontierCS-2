# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it computes
the quality metrics from the returned labels and centroids. The train judge's image leaves out the
hidden workloads' shapes and seed.
