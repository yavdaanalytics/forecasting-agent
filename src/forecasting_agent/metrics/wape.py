from __future__ import annotations

import math

import numpy as np


def wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Weighted Absolute Percentage Error: sum(|y - yhat|) / sum(|y|)."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    n = min(y.size, yhat.size)
    if n == 0:
        return None
    y = y[:n]
    yhat = yhat[:n]
    denom = float(np.sum(np.abs(y)))
    if math.isclose(denom, 0.0):
        return None
    return float(np.sum(np.abs(y - yhat)) / denom)
