from datetime import date, timedelta

from forecasting_agent.domain.types import SalesSeries


def make_series(
    sku: str,
    values: list[float],
    *,
    start: date | None = None,
    brand: str = "TAOS",
    category: str | None = None,
) -> SalesSeries:
    start = start or date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return SalesSeries.from_pairs(sku, dates, values, brand=brand, category=category)
