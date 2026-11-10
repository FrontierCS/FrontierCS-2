"""Starter: a straight line by least squares. Plain Python; PyTorch is there if you want it."""


def fit(xs: list[float], ys: list[float]) -> dict:
    """Fit the training points; return the model as a JSON-serialisable dict."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return {"slope": slope, "intercept": mean_y - slope * mean_x}


def predict(params: dict, xs: list[float]) -> list[float]:
    """The model's prediction at each x."""
    return [params["slope"] * x + params["intercept"] for x in xs]
