from __future__ import annotations

import math

import numpy as np

from forecasting_agent.domain.policy import Regime, SeriesDiagnosis
from forecasting_agent.domain.types import SalesSeries, SegmentAssignment

SHORT_HISTORY_POINTS = 60
INTERMITTENT_ZERO_FRAC = 0.6
STRUCTURAL_SHIFT_RATIO = 2.0
RECENT_WINDOW = 30


def zero_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return float(np.mean(arr == 0.0))


def recent_shift_ratio(values: np.ndarray, window: int = RECENT_WINDOW) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < window * 2:
        return None
    recent = float(np.mean(arr[-window:]))
    previous = float(np.mean(arr[-2 * window : -window]))
    if math.isclose(previous, 0.0):
        if math.isclose(recent, 0.0):
            return 1.0
        return None
    return recent / previous


def diagnose(series: SalesSeries, assignment: SegmentAssignment) -> SeriesDiagnosis:
    values = series.value_array
    zf = zero_fraction(values)
    shift = recent_shift_ratio(values)
    n = assignment.n_points
    regime, reason = _regime(assignment, n, zf, shift)
    return SeriesDiagnosis(
        sku=series.sku,
        regime=regime,
        n_points=n,
        zero_fraction=zf,
        recent_shift_ratio=shift,
        cv_segment=assignment.segment,
        changepoints_per_year=assignment.changepoints_per_year,
        reason=reason,
    )


def _regime(
    assignment: SegmentAssignment,
    n: int,
    zero_frac: float,
    shift: float | None,
) -> tuple[Regime, str]:
    if assignment.segment == "insufficient":
        return "insufficient", assignment.reason or "insufficient"
    if n < SHORT_HISTORY_POINTS:
        return "short_history", "too_few_points_for_holdout"
    if zero_frac >= INTERMITTENT_ZERO_FRAC:
        return "intermittent", "high_zero_fraction"
    if shift is not None and (shift >= STRUCTURAL_SHIFT_RATIO or shift <= 1.0 / STRUCTURAL_SHIFT_RATIO):
        return "structural_break", "recent_mean_shift"
    if assignment.segment == "stable":
        return "stable", "cv_stable"
    return "volatile", "cv_volatile"
