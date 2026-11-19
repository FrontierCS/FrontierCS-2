# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it
recomputes the statistics from the saved encoder weights. The train judge's image leaves out the
held-out seeds.
