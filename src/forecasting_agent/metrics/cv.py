from __future__ import annotations

import math

import numpy as np

from forecasting_agent.config.settings import CV_STABLE_THRESHOLD
from forecasting_agent.domain.types import SegmentAssignment, SegmentName


def coefficient_of_variation(values: np.ndarray) -> tuple[float | None, float, float]:
    """Return (cv, mean, std). cv is None when it is undefined."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if arr.size < 2:
        return None, mean, std
    if math.isclose(mean, 0.0):
        return None, mean, std
    return std / abs(mean), mean, std


def classify_cv(
    sku: str,
    values: np.ndarray,
    *,
    threshold: float = CV_STABLE_THRESHOLD,
) -> SegmentAssignment:
    cv, mean, std = coefficient_of_variation(values)
    n = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    if n < 2:
        return SegmentAssignment(
            sku=sku, cv=cv, mean=mean, std=std, n_points=n, segment="insufficient", reason="too_few_points"
        )
    if cv is None:
        return SegmentAssignment(
            sku=sku, cv=cv, mean=mean, std=std, n_points=n, segment="insufficient", reason="zero_mean"
        )
    segment: SegmentName = "stable" if cv < threshold else "volatile"
    return SegmentAssignment(sku=sku, cv=cv, mean=mean, std=std, n_points=n, segment=segment)
