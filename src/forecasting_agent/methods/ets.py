from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from forecasting_agent.config.settings import SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries
from forecasting_agent.methods.baseline import MovingAverageBaseline


class ETSMethod:
    """Holt-Winters (additive). Falls back to moving average if statsmodels is missing."""

    name = "ets"

    def fit_predict(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None = None,
    ) -> ForecastResult:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            return self._fallback(series, horizon=horizon, config=config)

        values = series.value_array
        if values.size < 2:
            return self._fallback(series, horizon=horizon, config=config)

        seasonal = values.size >= 14
        try:
            model = ExponentialSmoothing(
                values,
                trend="add",
                seasonal="add" if seasonal else None,
                seasonal_periods=7 if seasonal else None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            raw = np.asarray(fitted.forecast(horizon), dtype=float)
        except Exception:
            return self._fallback(series, horizon=horizon, config=config)

        preds = tuple(max(0.0, float(v)) for v in raw[:horizon])
        dates = _forecast_dates(series, horizon)
        return ForecastResult(
            sku=series.sku,
            method=self.name,
            dates=dates,
            values=preds,
            metadata={"seasonal": int(seasonal), "seasonal_periods": 7 if seasonal else 0},
        )

    def _fallback(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None,
    ) -> ForecastResult:
        fallback = MovingAverageBaseline().fit_predict(series, horizon=horizon, config=config)
        return ForecastResult(
            sku=fallback.sku,
            method=self.name,
            dates=fallback.dates,
            values=fallback.values,
            metadata={**dict(fallback.metadata), "fallback": "baseline"},
        )


def _forecast_dates(series: SalesSeries, horizon: int) -> tuple[date, ...]:
    start = series.dates[-1] + timedelta(days=1) if series.dates else date.today()
    return tuple(start + timedelta(days=i) for i in range(horizon))
