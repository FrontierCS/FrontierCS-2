# How this port differs

The judges' images hold the port's `eval/evaluator.py` in place of `judge/evaluator.py`; in the
final call it replays the predictor's logged answers in its own build (`eval/seer/`). The train
judge's image leaves out the blocks the train judge never replays.
