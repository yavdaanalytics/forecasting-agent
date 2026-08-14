from __future__ import annotations

from datetime import date, timedelta

from forecasting_agent.domain.types import SalesSeries


def zero_fill_daily(series: SalesSeries) -> SalesSeries:
    """Insert missing calendar days with quantity 0 so zeros are low demand, not gaps."""
    if len(series.dates) == 0:
        return series
    start = min(series.dates)
    end = max(series.dates)
    lookup = dict(zip(series.dates, series.values, strict=True))
    dates: list[date] = []
    values: list[float] = []
    day = start
    while day <= end:
        dates.append(day)
        values.append(float(lookup.get(day, 0.0)))
        day += timedelta(days=1)
    return SalesSeries(
        sku=series.sku,
        dates=tuple(dates),
        values=tuple(values),
        brand=series.brand,
        category=series.category,
    )
