from __future__ import annotations

import numpy as np

from forecasting_agent.config.settings import BrandConfig, SegmentConfig
from forecasting_agent.domain.inventory import SafetyStockResult
from forecasting_agent.domain.policy import MethodDecision, SeriesDiagnosis
from forecasting_agent.domain.types import SalesSeries
from forecasting_agent.inventory.backfill import lead_time_errors
from forecasting_agent.inventory.formula import ss_by_service_level, safety_stock_units, z_for_service_level
from forecasting_agent.methods.protocol import ForecastMethod

SKIP_REGIMES = frozenset({"insufficient", "short_history", "intermittent", "structural_break"})
MIN_ORIGINS = 3


def skipped_result(
    sku: str,
    method: str,
    *,
    lead_time_days: int,
    service_level: float,
    reason: str,
) -> SafetyStockResult:
    return SafetyStockResult(
        sku=sku,
        method=method,
        lead_time_days=lead_time_days,
        service_level=service_level,
        n_origins=0,
        error_std=None,
        error_bias=None,
        z=z_for_service_level(service_level),
        safety_stock=None,
        by_service_level={},
        reason=reason,
        escalate=True,
    )


def compute_safety_stock(
    series: SalesSeries,
    diagnosis: SeriesDiagnosis,
    decision: MethodDecision,
    method: ForecastMethod | None,
    brand: BrandConfig,
    config: SegmentConfig | None,
) -> SafetyStockResult:
    lead = brand.lead_time_days
    sl = brand.service_level
    chosen = decision.method

    if diagnosis.regime in SKIP_REGIMES:
        return skipped_result(
            series.sku, chosen, lead_time_days=lead, service_level=sl, reason=f"skip_{diagnosis.regime}"
        )
    if decision.escalate or chosen in {"none", ""}:
        return skipped_result(
            series.sku, chosen, lead_time_days=lead, service_level=sl, reason="skip_escalated_forecast"
        )
    if method is None:
        return skipped_result(
            series.sku, chosen, lead_time_days=lead, service_level=sl, reason="skip_missing_method"
        )

    errors = lead_time_errors(
        series,
        method,
        lead_time_days=lead,
        config=config,
        stride=brand.backfill_stride,
        max_origins=brand.backfill_max_origins,
    )
    if len(errors) < MIN_ORIGINS:
        return skipped_result(
            series.sku,
            chosen,
            lead_time_days=lead,
            service_level=sl,
            reason="skip_too_few_backfill_origins",
        )

    arr = np.asarray(errors, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    bias = float(np.mean(arr))
    ss = safety_stock_units(std, service_level=sl, error_bias=bias)
    return SafetyStockResult(
        sku=series.sku,
        method=chosen,
        lead_time_days=lead,
        service_level=sl,
        n_origins=len(errors),
        error_std=round(std, 4),
        error_bias=round(bias, 4),
        z=round(z_for_service_level(sl), 4),
        safety_stock=round(ss, 4),
        by_service_level=ss_by_service_level(std, bias),
        reason="rolling_origin_error",
        escalate=False,
    )
