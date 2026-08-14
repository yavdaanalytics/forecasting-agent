from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from forecasting_agent.domain.policy import MethodDecision, SeriesDiagnosis

import numpy as np

SegmentName = Literal["stable", "volatile", "insufficient"]
MethodName = str


def _as_date(value: date | datetime | np.datetime64) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return np.datetime64(value, "D").astype(datetime).date()


@dataclass(frozen=True)
class SalesSeries:
    sku: str
    dates: tuple[date, ...]
    values: tuple[float, ...]
    brand: str | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError(
                f"{self.sku}: dates ({len(self.dates)}) and values ({len(self.values)}) differ"
            )

    @classmethod
    def from_pairs(
        cls,
        sku: str,
        dates: Sequence[date | datetime | np.datetime64],
        values: Sequence[float],
        *,
        brand: str | None = None,
        category: str | None = None,
    ) -> SalesSeries:
        return cls(
            sku=sku,
            dates=tuple(_as_date(d) for d in dates),
            values=tuple(float(v) for v in values),
            brand=brand,
            category=category,
        )

    @property
    def value_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def slice_head(self, n: int) -> SalesSeries:
        return SalesSeries(
            sku=self.sku,
            dates=self.dates[:n],
            values=self.values[:n],
            brand=self.brand,
            category=self.category,
        )

    def slice_tail(self, n: int) -> SalesSeries:
        return SalesSeries(
            sku=self.sku,
            dates=self.dates[-n:],
            values=self.values[-n:],
            brand=self.brand,
            category=self.category,
        )


@dataclass(frozen=True)
class SegmentAssignment:
    sku: str
    cv: float | None
    mean: float
    std: float
    n_points: int
    segment: SegmentName
    reason: str | None = None
    n_changepoints: int = 0
    changepoints_per_year: float = 0.0


@dataclass(frozen=True)
class SegmentSummary:
    name: SegmentName
    skus: tuple[str, ...]
    avg_cv: float | None
    config: Mapping[str, float | int]
    avg_changepoints_per_year: float | None = None

    @property
    def count(self) -> int:
        return len(self.skus)

    def to_dict(self) -> dict:
        return {
            "skus": list(self.skus),
            "count": self.count,
            "avg_cv": self.avg_cv,
            "avg_changepoints_per_year": self.avg_changepoints_per_year,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class ForecastResult:
    sku: str
    method: MethodName
    dates: tuple[date, ...]
    values: tuple[float, ...]
    metadata: Mapping[str, float | int | str] = field(default_factory=dict)

    @property
    def value_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True)
class MethodScore:
    method: MethodName
    wape: float | None
    recommended: bool = False


@dataclass(frozen=True)
class PipelineResult:
    brand: str
    assignments: tuple[SegmentAssignment, ...]
    segments: Mapping[str, SegmentSummary]
    forecasts: tuple[ForecastResult, ...]
    scores: Mapping[str, tuple[MethodScore, ...]]
    recommendations: Mapping[str, Mapping[str, str | float | int | bool]]
    diagnoses: Mapping[str, SeriesDiagnosis] = field(default_factory=dict)
    decisions: tuple[MethodDecision, ...] = ()
