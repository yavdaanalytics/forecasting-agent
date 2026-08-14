from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Regime = Literal[
    "insufficient",
    "short_history",
    "intermittent",
    "structural_break",
    "stable",
    "volatile",
]


@dataclass(frozen=True)
class SeriesDiagnosis:
    sku: str
    regime: Regime
    n_points: int
    zero_fraction: float
    recent_shift_ratio: float | None
    cv_segment: str
    changepoints_per_year: float
    reason: str


@dataclass(frozen=True)
class MethodDecision:
    sku: str
    method: str
    reason: str
    confidence: str
    escalate: bool
    wape: float | None
    candidates: tuple[str, ...]
