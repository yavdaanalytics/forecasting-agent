from __future__ import annotations

import math

import numpy as np

MIN_SIDE = 14
MAX_CHANGEPOINTS = 8
SHIFT_K = 1.5
ROLLING_WINDOW = 7


def detect_changepoints(
    values: np.ndarray,
    *,
    min_side: int = MIN_SIDE,
    k: float = SHIFT_K,
    max_points: int = MAX_CHANGEPOINTS,
    window: int = ROLLING_WINDOW,
) -> tuple[int, float]:
    """Binary-split mean-shift detector on a rolling mean.

    Returns (n_changepoints, changepoints_per_year).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_days = int(arr.size)
    if n_days < 2 * min_side:
        return 0, 0.0
    smooth = _rolling_mean(arr, window)
    found: list[int] = []
    _split_recurse(smooth, 0, min_side, k, max_points, found)
    n_cp = len(found)
    years = n_days / 365.25
    per_year = n_cp / years if years > 0 else 0.0
    return n_cp, float(per_year)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    w = min(window, int(arr.size))
    if w <= 1:
        return arr.copy()
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def _best_split(y: np.ndarray, min_side: int, k: float) -> int | None:
    n = int(y.size)
    if n < 2 * min_side:
        return None
    best_i: int | None = None
    best_score = 0.0
    for i in range(min_side, n - min_side + 1):
        left, right = y[:i], y[i:]
        ml, mr = float(np.mean(left)), float(np.mean(right))
        nl, nr = left.size, right.size
        var_l = float(np.var(left, ddof=0))
        var_r = float(np.var(right, ddof=0))
        pooled_var = ((nl - 1) * var_l + (nr - 1) * var_r) / max(nl + nr - 2, 1)
        pooled = math.sqrt(pooled_var) if pooled_var > 0 else float(np.std(y, ddof=0))
        if pooled < 1e-12:
            pooled = 1e-12
        score = abs(ml - mr) / pooled
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is not None and best_score > k:
        return best_i
    return None


def _split_recurse(
    y: np.ndarray,
    offset: int,
    min_side: int,
    k: float,
    cap: int,
    found: list[int],
) -> None:
    if len(found) >= cap:
        return
    split = _best_split(y, min_side, k)
    if split is None:
        return
    found.append(offset + split)
    _split_recurse(y[:split], offset, min_side, k, cap, found)
    _split_recurse(y[split:], offset + split, min_side, k, cap, found)
