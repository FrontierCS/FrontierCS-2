# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it
recomputes the results' inner products (`eval/inner_products.cpp`). The train judge's image leaves
out the final-only heap capacities and the ground truth of the second split's queries.
