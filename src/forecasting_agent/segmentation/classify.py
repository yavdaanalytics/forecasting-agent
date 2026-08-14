from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from forecasting_agent.config.settings import BrandConfig, CV_STABLE_THRESHOLD, segment_config_for
from forecasting_agent.domain.types import SalesSeries, SegmentAssignment, SegmentSummary
from forecasting_agent.metrics.changepoints import detect_changepoints
from forecasting_agent.metrics.cv import classify_cv


def assign_segments(
    series_list: list[SalesSeries],
    *,
    threshold: float = CV_STABLE_THRESHOLD,
) -> list[SegmentAssignment]:
    assignments: list[SegmentAssignment] = []
    for series in series_list:
        assignment = classify_cv(series.sku, series.value_array, threshold=threshold)
        n_cp, cp_year = detect_changepoints(series.value_array)
        assignments.append(
            replace(assignment, n_changepoints=n_cp, changepoints_per_year=cp_year)
        )
    return assignments


def summarize_segments(
    assignments: list[SegmentAssignment],
    brand: BrandConfig,
) -> dict[str, SegmentSummary]:
    buckets: dict[str, list[SegmentAssignment]] = defaultdict(list)
    for item in assignments:
        buckets[item.segment].append(item)
    summaries: dict[str, SegmentSummary] = {}
    for name, items in buckets.items():
        cvs = [a.cv for a in items if a.cv is not None]
        avg_cv = sum(cvs) / len(cvs) if cvs else None
        cp_years = [a.changepoints_per_year for a in items]
        avg_cp = sum(cp_years) / len(cp_years) if cp_years else None
        if name in ("stable", "volatile"):
            cfg = segment_config_for(brand, name)
            config = {
                "changepoint_prior_scale": cfg.changepoint_prior_scale,
                "horizon": cfg.horizon,
            }
        else:
            config = {"changepoint_prior_scale": 0.0, "horizon": 0}
        summaries[name] = SegmentSummary(
            name=name,  # type: ignore[arg-type]
            skus=tuple(a.sku for a in items),
            avg_cv=avg_cv,
            config=config,
            avg_changepoints_per_year=avg_cp,
        )
    return summaries
