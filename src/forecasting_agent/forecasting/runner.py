from __future__ import annotations

import asyncio
from typing import Sequence

from forecasting_agent.config.settings import SegmentConfig
from forecasting_agent.domain.types import ForecastResult, SalesSeries
from forecasting_agent.methods.protocol import ForecastMethod


async def forecast_series(
    series: SalesSeries,
    methods: Sequence[ForecastMethod],
    *,
    horizon: int,
    config: SegmentConfig | None = None,
) -> list[ForecastResult]:
    async def _one(method: ForecastMethod) -> ForecastResult:
        return await asyncio.to_thread(
            method.fit_predict, series, horizon=horizon, config=config
        )

    gathered = await asyncio.gather(*[_one(m) for m in methods], return_exceptions=True)
    results: list[ForecastResult] = []
    for method, item in zip(methods, gathered, strict=True):
        if isinstance(item, Exception):
            continue
        results.append(item)
    return results


def forecast_series_sync(
    series: SalesSeries,
    methods: Sequence[ForecastMethod],
    *,
    horizon: int,
    config: SegmentConfig | None = None,
) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    for method in methods:
        try:
            results.append(method.fit_predict(series, horizon=horizon, config=config))
        except Exception:
            continue
    return results
