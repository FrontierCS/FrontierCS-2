# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it takes a
testbench's PASS line only with a random nonce. The train judge's image leaves out the held-out
cases.
