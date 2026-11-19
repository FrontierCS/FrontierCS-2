# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; in the
final call it replays the predictor's recorded directions in its own simulator. The train judge's
image leaves out the 81 final traces, and the final rows of the reference table. A value or a line
of the judge also differs: the final call's timeout 12,000 s, not 7,200 s.
