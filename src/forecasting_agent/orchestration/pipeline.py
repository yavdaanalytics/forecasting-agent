from __future__ import annotations

from collections import defaultdict

from forecasting_agent.config.settings import BrandConfig, get_brand_config, tune_segment_config
from forecasting_agent.connectors.csv_store import MemorySalesStore
from forecasting_agent.connectors.protocols import SalesStore
from forecasting_agent.domain.policy import MethodDecision, SeriesDiagnosis
from forecasting_agent.domain.types import (
    ForecastResult,
    MethodScore,
    PipelineResult,
    SalesSeries,
    SegmentAssignment,
)
from forecasting_agent.forecasting.runner import forecast_series_sync
from forecasting_agent.methods.factory import build_methods
from forecasting_agent.methods.protocol import ForecastMethod
from forecasting_agent.inventory.compute import compute_safety_stock
from forecasting_agent.policy.candidates import candidate_names, filter_catalog
from forecasting_agent.policy.diagnose import diagnose
from forecasting_agent.policy.select import select_method
from forecasting_agent.recommendations.rank import recommend_from_decisions, rank_methods
from forecasting_agent.segmentation.classify import assign_segments, summarize_segments
from forecasting_agent.validation.holdout import score_holdout, split_holdout


class ForecastPipeline:
    """Diagnose → candidate methods → score → decide (WAPE or CV prior)."""

    def __init__(
        self,
        store: SalesStore,
        *,
        brand: str = "TAOS",
        methods: list[ForecastMethod] | None = None,
        validate: bool = True,
    ) -> None:
        self.store = store
        self.brand_name = brand
        self.brand: BrandConfig = get_brand_config(brand)
        self.methods = methods or build_methods(baseline_window=self.brand.baseline_window)
        self.validate = validate

    def run(self, brand: str | None = None) -> PipelineResult:
        series_list = list(self.store.load(brand or self.brand_name))
        assignments = assign_segments(series_list)
        by_sku = {s.sku: s for s in series_list}
        segments = summarize_segments(assignments, self.brand)
        forecasts: list[ForecastResult] = []
        diagnoses: dict[str, SeriesDiagnosis] = {}
        decisions: list[MethodDecision] = []
        sku_scores: dict[str, dict[str, float | None]] = {}

        for assignment in assignments:
            series = by_sku[assignment.sku]
            diag = diagnose(series, assignment)
            diagnoses[series.sku] = diag
            names = candidate_names(diag, self.methods)
            cands = filter_catalog(diag, self.methods)
            if not cands:
                decisions.append(select_method(diag, {}, names))
                continue
            cfg = tune_segment_config(
                self.brand, assignment.segment, assignment.changepoints_per_year
            )
            forecasts.extend(
                forecast_series_sync(series, cands, horizon=cfg.horizon, config=cfg)
            )
            scores_map: dict[str, float | None] = {}
            if self.validate:
                scores_map = self._score_sku(series, cands, cfg)
            sku_scores[series.sku] = scores_map
            decisions.append(select_method(diag, scores_map, names))

        scores = self._segment_scores(assignments, sku_scores)
        recommendations = {}
        for name, summary in segments.items():
            if name == "insufficient":
                continue
            sku_set = set(summary.skus)
            cfg = tune_segment_config(self.brand, name, summary.avg_changepoints_per_year or 0.0)
            recommendations[name] = recommend_from_decisions(
                name,
                [d for d in decisions if d.sku in sku_set],
                horizon=cfg.horizon,
                num_skus=summary.count,
            )

        by_name = {m.name: m for m in self.methods}
        safety = []
        if self.validate:
            for assignment in assignments:
                series = by_sku[assignment.sku]
                diag = diagnoses[series.sku]
                decision = next(d for d in decisions if d.sku == series.sku)
                cfg = tune_segment_config(
                    self.brand, assignment.segment, assignment.changepoints_per_year
                )
                safety.append(
                    compute_safety_stock(
                        series,
                        diag,
                        decision,
                        by_name.get(decision.method),
                        self.brand,
                        cfg,
                    )
                )

        return PipelineResult(
            brand=self.brand.name,
            assignments=tuple(assignments),
            segments=segments,
            forecasts=tuple(forecasts),
            scores=scores,
            recommendations=recommendations,
            diagnoses=diagnoses,
            decisions=tuple(decisions),
            safety_stock=tuple(safety),
        )

    def _score_sku(self, series: SalesSeries, methods: list[ForecastMethod], cfg) -> dict[str, float | None]:
        holdout_days = self.brand.holdout_days
        out: dict[str, float | None] = {m.name: None for m in methods}
        if len(series.dates) <= holdout_days:
            return out
        try:
            train, holdout = split_holdout(series, holdout_days)
        except ValueError:
            return out
        for method in methods:
            try:
                _, score = score_holdout(train, holdout, method, config=cfg)
            except Exception:
                continue
            out[method.name] = score
        return out

    def _segment_scores(
        self,
        assignments: list[SegmentAssignment],
        sku_scores: dict[str, dict[str, float | None]],
    ) -> dict[str, tuple[MethodScore, ...]]:
        per_segment: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for assignment in assignments:
            for name, wape in sku_scores.get(assignment.sku, {}).items():
                if wape is not None:
                    per_segment[assignment.segment][name].append(wape)
        ranked: dict[str, tuple[MethodScore, ...]] = {}
        for segment, method_scores in per_segment.items():
            averages = {
                name: (sum(vals) / len(vals) if vals else None)
                for name, vals in method_scores.items()
            }
            ranked[segment] = tuple(rank_methods(averages))
        return ranked


def run_from_series(series: list[SalesSeries], brand: str = "TAOS") -> PipelineResult:
    return ForecastPipeline(MemorySalesStore(series), brand=brand).run()
