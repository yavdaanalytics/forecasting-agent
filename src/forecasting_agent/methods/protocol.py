from __future__ import annotations

from typing import Protocol, runtime_checkable

from forecasting_agent.config.settings import SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries


@runtime_checkable
class ForecastMethod(Protocol):
    name: str

    def fit_predict(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None = None,
    ) -> ForecastResult:
        ...
