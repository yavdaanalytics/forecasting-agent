from __future__ import annotations

from forecasting_agent.config.settings import CHANGEPOINTS_FEW_PER_YEAR
from forecasting_agent.domain.policy import MethodDecision, SeriesDiagnosis
from forecasting_agent.recommendations.rank import confidence_label

CLEAR_WAPE_GAP = 0.05
ESCALATE_WAPE = 0.50


def cv_prior_method(diagnosis: SeriesDiagnosis, candidates: tuple[str, ...]) -> str:
    """Prior when holdout is messy or missing."""
    if diagnosis.regime in ("short_history", "intermittent", "structural_break"):
        preferred = "baseline"
    elif diagnosis.cv_segment == "stable":
        preferred = "prophet" if diagnosis.changepoints_per_year < CHANGEPOINTS_FEW_PER_YEAR else "ensemble_ets"
    else:
        preferred = (
            "ensemble_ets" if diagnosis.changepoints_per_year >= CHANGEPOINTS_FEW_PER_YEAR else "baseline"
        )
    if preferred in candidates:
        return preferred
    if "baseline" in candidates:
        return "baseline"
    return candidates[0] if candidates else "baseline"


def select_method(
    diagnosis: SeriesDiagnosis,
    scores: dict[str, float | None],
    candidates: tuple[str, ...],
) -> MethodDecision:
    if diagnosis.regime == "insufficient" or not candidates:
        return MethodDecision(
            sku=diagnosis.sku,
            method="none",
            reason="escalate_insufficient",
            confidence="low",
            escalate=True,
            wape=None,
            candidates=candidates,
        )

    valid = {name: wape for name, wape in scores.items() if wape is not None and name in candidates}
    if not valid:
        prior = cv_prior_method(diagnosis, candidates)
        return MethodDecision(
            sku=diagnosis.sku,
            method=prior,
            reason="cv_prior_no_holdout",
            confidence="low",
            escalate=True,
            wape=None,
            candidates=candidates,
        )

    ranked = sorted(valid.items(), key=lambda kv: kv[1])
    best_name, best_wape = ranked[0]

    if best_wape >= ESCALATE_WAPE:
        return MethodDecision(
            sku=diagnosis.sku,
            method=best_name,
            reason="high_wape_review",
            confidence="low",
            escalate=True,
            wape=best_wape,
            candidates=candidates,
        )

    if len(ranked) >= 2 and (ranked[1][1] - best_wape) < CLEAR_WAPE_GAP:
        prior = cv_prior_method(diagnosis, candidates)
        chosen = prior if prior in valid else best_name
        return MethodDecision(
            sku=diagnosis.sku,
            method=chosen,
            reason="cv_prior_close_wape",
            confidence="medium",
            escalate=False,
            wape=valid.get(chosen, best_wape),
            candidates=candidates,
        )

    return MethodDecision(
        sku=diagnosis.sku,
        method=best_name,
        reason="holdout_wape",
        confidence=confidence_label(best_wape),
        escalate=False,
        wape=best_wape,
        candidates=candidates,
    )
