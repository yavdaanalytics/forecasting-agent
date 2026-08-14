from __future__ import annotations

from datetime import timedelta

import numpy as np

from forecasting_agent.config.settings import DEFAULT_BASELINE_WINDOW, SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries


class MovingAverageBaseline:
    def __init__(self, window: int = DEFAULT_BASELINE_WINDOW, *, name: str = "baseline") -> None:
        self.window = window
        self.name = name

    def fit_predict(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None = None,
    ) -> ForecastResult:
        values = series.value_array
        if values.size == 0:
            level = 0.0
        else:
            w = min(self.window, values.size)
            level = float(np.mean(values[-w:]))
        start = series.dates[-1] + timedelta(days=1) if series.dates else None
        if start is None:
            from datetime import date

            start = date.today()
        dates = tuple(start + timedelta(days=i) for i in range(horizon))
        preds = tuple(level for _ in range(horizon))
        return ForecastResult(
            sku=series.sku,
            method=self.name,
            dates=dates,
            values=preds,
            metadata={"window": w if values.size else 0, "level": level},
        )
