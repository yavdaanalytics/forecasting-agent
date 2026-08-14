from datetime import date, timedelta

from forecasting_agent.domain.calendar import zero_fill_daily
from forecasting_agent.domain.types import SalesSeries


def test_zero_fill_inserts_missing_days():
    series = SalesSeries.from_pairs(
        "SKU",
        [date(2024, 1, 1), date(2024, 1, 3)],
        [10.0, 12.0],
    )
    filled = zero_fill_daily(series)
    assert len(filled.dates) == 3
    assert filled.values == (10.0, 0.0, 12.0)
    assert filled.dates[1] == date(2024, 1, 1) + timedelta(days=1)
