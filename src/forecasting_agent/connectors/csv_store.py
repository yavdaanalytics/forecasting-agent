from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from forecasting_agent.domain.calendar import zero_fill_daily
from forecasting_agent.domain.types import SalesSeries


class CsvSalesStore:
    """Load sku,date,qty[,brand,category] from a CSV file."""

    def __init__(self, path: str | Path, *, fill_zeros: bool = True) -> None:
        self.path = Path(path)
        self.fill_zeros = fill_zeros

    def load(self, brand: str | None = None) -> Sequence[SalesSeries]:
        frame = pd.read_csv(self.path, parse_dates=["date"])
        required = {"sku", "date", "qty"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.path}: missing columns {sorted(missing)}")
        if brand is not None and "brand" in frame.columns:
            frame = frame[frame["brand"].astype(str).str.upper() == brand.upper()]
        grouped: dict[str, list[tuple[date, float, str | None, str | None]]] = defaultdict(list)
        has_brand = "brand" in frame.columns
        has_cat = "category" in frame.columns
        for row in frame.itertuples(index=False):
            sku = str(row.sku)
            d = row.date.date() if hasattr(row.date, "date") else date.fromisoformat(str(row.date)[:10])
            b = str(row.brand) if has_brand else brand
            cat = str(row.category) if has_cat else None
            grouped[sku].append((d, float(row.qty), b, cat))
        series_list: list[SalesSeries] = []
        for sku, pairs in grouped.items():
            pairs.sort(key=lambda item: item[0])
            item = SalesSeries.from_pairs(
                sku,
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                brand=pairs[0][2],
                category=pairs[0][3],
            )
            if self.fill_zeros:
                item = zero_fill_daily(item)
            series_list.append(item)
        return series_list


class MemorySalesStore:
    def __init__(self, series: Sequence[SalesSeries]) -> None:
        self._series = list(series)

    def load(self, brand: str | None = None) -> Sequence[SalesSeries]:
        if brand is None:
            return list(self._series)
        key = brand.upper()
        return [s for s in self._series if (s.brand or "").upper() == key]
