from __future__ import annotations

import math

from forecasting_agent.domain.inventory import Z_BY_SERVICE_LEVEL


def z_for_service_level(service_level: float) -> float:
    """Normal z for cycle service level. Known table, else inverse-erf approx."""
    key = round(float(service_level), 2)
    if key in Z_BY_SERVICE_LEVEL:
        return Z_BY_SERVICE_LEVEL[key]
    p = min(0.999, max(0.5, float(service_level)))
    return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)


def safety_stock_units(
    error_std: float,
    *,
    service_level: float,
    error_bias: float = 0.0,
) -> float:
    """SS = z*σ + max(0, bias). Bias is mean(actual - forecast) over lead time."""
    z = z_for_service_level(service_level)
    std = max(0.0, float(error_std))
    extra = max(0.0, float(error_bias))
    return max(0.0, z * std + extra)


def ss_by_service_level(error_std: float, error_bias: float = 0.0) -> dict[str, float]:
    return {
        f"{lvl:.2f}": round(safety_stock_units(error_std, service_level=lvl, error_bias=error_bias), 4)
        for lvl in (0.90, 0.95, 0.99)
    }
