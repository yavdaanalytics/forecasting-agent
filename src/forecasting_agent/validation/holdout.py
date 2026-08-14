from __future__ import annotations

from forecasting_agent.domain.types import ForecastResult, SalesSeries
from forecasting_agent.metrics.wape import wape
from forecasting_agent.methods.protocol import ForecastMethod


def split_holdout(series: SalesSeries, holdout_days: int) -> tuple[SalesSeries, SalesSeries]:
    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive")
    if len(series.dates) <= holdout_days:
        raise ValueError(f"{series.sku}: history shorter than holdout ({holdout_days})")
    train_n = len(series.dates) - holdout_days
    return series.slice_head(train_n), series.slice_tail(holdout_days)


def score_holdout(
    train: SalesSeries,
    holdout: SalesSeries,
    method: ForecastMethod,
    *,
    config=None,
) -> tuple[ForecastResult, float | None]:
    forecast = method.fit_predict(train, horizon=len(holdout.dates), config=config)
    score = wape(holdout.value_array, forecast.value_array)
    return forecast, score
