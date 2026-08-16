from __future__ import annotations

from forecasting_agent.domain.types import PipelineResult


def accuracy_table(result: PipelineResult) -> dict:
    out: dict = {}
    for segment, scores in result.scores.items():
        row = {s.method + "_wape": s.wape for s in scores}
        winner = next((s.method for s in scores if s.recommended), None)
        row["recommended_method"] = winner
        out[segment] = row
    return out


def pipeline_as_dict(result: PipelineResult) -> dict:
    return {
        "brand": result.brand,
        "segments": {name: summary.to_dict() for name, summary in result.segments.items()},
        "accuracy": accuracy_table(result),
        "recommendations": dict(result.recommendations),
        "forecast_count": len(result.forecasts),
        "decisions": [
            {
                "sku": d.sku,
                "method": d.method,
                "reason": d.reason,
                "confidence": d.confidence,
                "escalate": d.escalate,
                "wape": d.wape,
                "candidates": list(d.candidates),
            }
            for d in result.decisions
        ],
        "diagnoses": {
            sku: {
                "regime": diag.regime,
                "zero_fraction": diag.zero_fraction,
                "changepoints_per_year": diag.changepoints_per_year,
                "reason": diag.reason,
            }
            for sku, diag in result.diagnoses.items()
        },
        "safety_stock": [s.to_dict() for s in result.safety_stock],
    }
