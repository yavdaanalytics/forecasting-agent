from __future__ import annotations

from forecasting_agent.config.settings import SegmentConfig
from forecasting_agent.domain.types import SalesSeries
from forecasting_agent.methods.protocol import ForecastMethod


def lead_time_errors(
    series: SalesSeries,
    method: ForecastMethod,
    *,
    lead_time_days: int,
    config: SegmentConfig | None = None,
    stride: int = 7,
    max_origins: int = 6,
    min_train: int = 40,
) -> list[float]:
    """Rolling-origin lead-time errors: sum(actual) - sum(forecast) over L days.

    Out-of-sample only: each origin uses history strictly before the window.
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be positive")
    n = len(series.values)
    last_origin = n - lead_time_days
    if last_origin <= min_train:
        return []
    origins = list(range(min_train, last_origin + 1, max(1, stride)))
    origins = origins[-max_origins:]
    errors: list[float] = []
    for origin in origins:
        train = series.slice_head(origin)
        actual_sum = float(sum(series.values[origin : origin + lead_time_days]))
        try:
            forecast = method.fit_predict(train, horizon=lead_time_days, config=config)
        except Exception:
            continue
        pred_sum = float(sum(forecast.values[:lead_time_days]))
        errors.append(actual_sum - pred_sum)
    return errors
