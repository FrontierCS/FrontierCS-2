"""Fits and scores the model on a synthetic curve like the judge's (not one of its workloads).

    python3 run_local.py
"""
import math
import random

import model

rng = random.Random(7)
xs = [rng.uniform(-1.0, 1.0) for _ in range(2000)]
ys = [0.8 * math.sin(3.0 * x + 0.4) + 0.5 * x * x + rng.gauss(0.0, 0.1) for x in xs]
queries = [rng.uniform(-1.0, 1.0) for _ in range(500)]
truth = [0.8 * math.sin(3.0 * x + 0.4) + 0.5 * x * x for x in queries]
predictions = model.predict(model.fit(xs, ys), queries)
print({"mse": sum((p - t) ** 2 for p, t in zip(predictions, truth)) / len(truth)})
