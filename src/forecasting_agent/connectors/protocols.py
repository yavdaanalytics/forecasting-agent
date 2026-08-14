from __future__ import annotations

from typing import Protocol, Sequence

from forecasting_agent.domain.types import SalesSeries


class SalesStore(Protocol):
    """Read historical demand. Implementations: CSV, memory, BigQuery."""

    def load(self, brand: str | None = None) -> Sequence[SalesSeries]:
        ...


class ForecastClient(Protocol):
    """Remote or local execution of a named forecast method."""

    def forecast(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        method: str,
        changepoint_prior_scale: float,
        seasonality_prior_scale: float,
        weekly_seasonality: bool,
        yearly_seasonality: bool,
        daily_seasonality: bool,
    ) -> tuple[float, ...]:
        ...
