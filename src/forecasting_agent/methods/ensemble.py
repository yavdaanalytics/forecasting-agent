from __future__ import annotations

from forecasting_agent.config.settings import DEFAULT_ENSEMBLE_PROPHET_WEIGHT, SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries
from forecasting_agent.methods.protocol import ForecastMethod


class EnsembleMethod:
    def __init__(
        self,
        primary: ForecastMethod,
        secondary: ForecastMethod,
        *,
        primary_weight: float | None = None,
        name: str = "ensemble",
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_weight = primary_weight
        self.name = name

    def fit_predict(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None = None,
    ) -> ForecastResult:
        weight = (
            self.primary_weight
            if self.primary_weight is not None
            else (config.prophet_weight if config else DEFAULT_ENSEMBLE_PROPHET_WEIGHT)
        )
        weight = min(1.0, max(0.0, weight))
        a = self.primary.fit_predict(series, horizon=horizon, config=config)
        b = self.secondary.fit_predict(series, horizon=horizon, config=config)
        n = min(len(a.values), len(b.values), horizon)
        blended = tuple(weight * a.values[i] + (1.0 - weight) * b.values[i] for i in range(n))
        return ForecastResult(
            sku=series.sku,
            method=self.name,
            dates=a.dates[:n],
            values=blended,
            metadata={
                "primary": self.primary.name,
                "secondary": self.secondary.name,
                "primary_weight": weight,
            },
        )
