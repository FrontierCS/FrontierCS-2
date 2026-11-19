# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; it replays
the policy's rounds in its own simulator. The train judge's image leaves out the final suite's two
traces.
