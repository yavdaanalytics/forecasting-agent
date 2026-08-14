from __future__ import annotations

from collections import Counter

from forecasting_agent.domain.policy import MethodDecision
from forecasting_agent.domain.types import MethodScore

HIGH_WAPE = 0.40
MEDIUM_WAPE = 0.50


def confidence_label(wape: float | None) -> str:
    if wape is None:
        return "unknown"
    if wape < HIGH_WAPE:
        return "high"
    if wape <= MEDIUM_WAPE:
        return "medium"
    return "low"


def rank_methods(scores: dict[str, float | None]) -> list[MethodScore]:
    ranked = sorted(
        scores.items(),
        key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0.0),
    )
    best = ranked[0][0] if ranked and ranked[0][1] is not None else None
    return [
        MethodScore(method=name, wape=value, recommended=(name == best))
        for name, value in ranked
    ]


def recommend_segment(
    segment: str,
    scores: dict[str, float | None],
    *,
    horizon: int,
    num_skus: int,
) -> dict[str, str | float | int]:
    ranked = rank_methods(scores)
    winner = next((s for s in ranked if s.recommended), ranked[0] if ranked else None)
    wape_val = winner.wape if winner else None
    return {
        "segment": segment,
        "method": winner.method if winner else "prophet",
        "wape": round(wape_val, 4) if wape_val is not None else "",
        "horizon_days": horizon,
        "confidence": confidence_label(wape_val),
        "num_skus": num_skus,
        "selection_reason": "holdout_wape",
        "escalate": False,
    }


def recommend_from_decisions(
    segment: str,
    decisions: list[MethodDecision],
    *,
    horizon: int,
    num_skus: int,
) -> dict[str, str | float | int | bool]:
    usable = [d for d in decisions if d.method not in {"none", ""}]
    if not usable:
        rec = recommend_segment(segment, {}, horizon=horizon, num_skus=num_skus)
        rec["selection_reason"] = "escalate_insufficient"
        rec["escalate"] = True
        rec["method"] = "none"
        return rec
    winner = Counter(d.method for d in decisions if d.method not in {"none", ""}).most_common(1)[0][0]
    wapes = [d.wape for d in usable if d.method == winner and d.wape is not None]
    wape_val = sum(wapes) / len(wapes) if wapes else None
    reasons = [d.reason for d in usable if d.method == winner]
    reason = Counter(reasons).most_common(1)[0][0] if reasons else "holdout_wape"
    return {
        "segment": segment,
        "method": winner,
        "wape": round(wape_val, 4) if wape_val is not None else "",
        "horizon_days": horizon,
        "confidence": confidence_label(wape_val) if wape_val is not None else "low",
        "num_skus": num_skus,
        "selection_reason": reason,
        "escalate": any(d.escalate for d in usable),
    }
