"""Generate a small TAOS-like history CSV for local runs."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START = date(2024, 1, 1)
DAYS = 400


def main() -> None:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    specs = [
        ("PLANT-001", "plants", "stable", 10.0, 0.8),
        ("PLANT-002", "plants", "stable", 8.0, 0.6),
        ("SEED-101", "seeds", "stable", 4.0, 0.4),
        ("PLANT-301", "plants", "volatile", 3.0, 4.5),
        ("PLANT-302", "plants", "volatile", 2.0, 5.0),
        ("SOIL-201", "soil", "volatile", 1.5, 3.0),
    ]
    for sku, category, kind, mean, noise in specs:
        for i in range(DAYS):
            d = START + timedelta(days=i)
            if kind == "stable":
                qty = max(0.0, float(mean + rng.normal(0, noise)))
            else:
                qty = float(rng.choice([0.0, 0.0, 0.0, mean, mean * 6], p=[0.4, 0.2, 0.15, 0.15, 0.1]))
            rows.append(
                {
                    "sku": sku,
                    "date": d.isoformat(),
                    "qty": round(qty, 2),
                    "brand": "TAOS",
                    "category": category,
                }
            )
    out = Path(__file__).with_name("sample_taos_input.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
