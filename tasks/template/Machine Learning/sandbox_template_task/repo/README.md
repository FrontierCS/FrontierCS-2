# Curve Fitting on Sandboxes

Fit noisy samples of a smooth curve so that your predictions at new points are as close to the
true curve as possible.

## Task
`model.fit(xs, ys)` receives training points and returns a model; `model.predict(params, xs)`
evaluates it at query points. The starter fits a straight line: it runs and is poor.

## What you may change
- Only files below `model/`, at most 10 of them; new modules are allowed. `run_local.py` is
  fixed.
- `fit` must learn from its arguments only: reading files, using the network, embedding curve
  parameters and detecting the judge's inputs are forbidden. PyTorch is available on the GPU
  machine your code runs on; nothing else may be installed.

## Interface
`import model` must expose `fit(xs: list[float], ys: list[float]) -> dict` and
`predict(params: dict, xs: list[float]) -> list[float]`. `params` must be JSON-serialisable (at
most 1 MiB as JSON): the judge stores it and passes it to `predict` in a separate process.
`predict` returns one finite number per query.

## Correctness
`fit` and `predict` for all workloads run in one process on a machine of their own with an
H100, within 300 seconds. A crash, a timeout, a missing or malformed result, or a predict that
fails or exceeds 60 seconds on the timing machine scores the whole submission 0.

## Workload
One workload is one curve `y = a * sin(w * x + p) + q * x^2` with `a` in `[0.5, 1.2]`, `w` in
`[3, 8]`, `p` in `[-1.5, 1.5]` and `q` in `[-1, 1]`: 2,000 training points with `x` uniform in
`[-1, 1]` and Gaussian noise of standard deviation 0.1 on `y`, and 500 query points from the
same range. Train measures three public curves; final measures those three plus two held-out
curves. Your code and the judge's reference fit the same data, each on its own machine. Then one
trusted machine runs each `predict` three times per workload, interleaved with the reference's,
each in a fresh process.

## Scoring

```text
score = clamp(50 + 25 * log2(G), 0, 100)
G     = geometric mean over workloads of (reference MSE / your MSE)
```

MSE is the mean squared error of your 500 predictions against the noise-free curve. 50 at the
reference's error, 75 at half of it, 100 at a quarter.

**Beats the reference**: on every workload, mse must be strictly less than the reference's
(zero tolerance) and predict_seconds (the smallest of three run times of a fresh predict
process, from start to exit) at most the reference's plus 1.0 second.

## Running locally

```sh
python3 run_local.py     # fits a sample curve and prints the MSE of the predictions
```
