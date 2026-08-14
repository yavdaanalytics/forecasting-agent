from __future__ import annotations

from datetime import timedelta

import pandas as pd

from forecasting_agent.config.settings import STABLE_DEFAULT, SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries
from forecasting_agent.methods.baseline import MovingAverageBaseline


class ProphetUnavailableError(RuntimeError):
    pass


def _import_prophet():
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ProphetUnavailableError(
            "prophet is not installed. pip install 'forecasting-agent[prophet]'"
        ) from exc
    return Prophet


class ProphetMethod:
    name = "prophet"

    def fit_predict(
        self,
        series: SalesSeries,
        *,
        horizon: int,
        config: SegmentConfig | None = None,
    ) -> ForecastResult:
        cfg = config or STABLE_DEFAULT
        Prophet = _import_prophet()
        if len(series.dates) < 2:
            fallback = MovingAverageBaseline().fit_predict(series, horizon=horizon, config=cfg)
            return ForecastResult(
                sku=fallback.sku,
                method=self.name,
                dates=fallback.dates,
                values=fallback.values,
                metadata={**dict(fallback.metadata), "fallback": "baseline"},
            )
        frame = pd.DataFrame({"ds": pd.to_datetime(list(series.dates)), "y": list(series.values)})
        model = Prophet(
            changepoint_prior_scale=cfg.changepoint_prior_scale,
            seasonality_prior_scale=cfg.seasonality_prior_scale,
            weekly_seasonality=cfg.weekly_seasonality,
            yearly_seasonality=cfg.yearly_seasonality,
            daily_seasonality=cfg.daily_seasonality,
        )
        model.fit(frame)
        future = model.make_future_dataframe(periods=horizon, include_history=False)
        forecast = model.predict(future)
        dates = tuple((series.dates[-1] + timedelta(days=i + 1)) for i in range(horizon))
        values = tuple(float(v) for v in forecast["yhat"].tolist()[:horizon])
        return ForecastResult(
            sku=series.sku,
            method=self.name,
            dates=dates,
            values=values,
            metadata={
                "changepoint_prior_scale": cfg.changepoint_prior_scale,
                "seasonality_prior_scale": cfg.seasonality_prior_scale,
            },
        )
