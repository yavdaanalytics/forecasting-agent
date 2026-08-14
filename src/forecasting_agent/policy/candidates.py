from __future__ import annotations

from forecasting_agent.domain.policy import SeriesDiagnosis
from forecasting_agent.methods.protocol import ForecastMethod

# Hard regimes: cheap, robust methods. Healthy series: full catalog.
_REGIME_NAMES: dict[str, tuple[str, ...]] = {
    "insufficient": (),
    "short_history": ("baseline", "ets"),
    "intermittent": ("baseline", "ets"),
    "structural_break": ("baseline", "ets", "ensemble_ma", "ensemble_ets"),
}


def candidate_names(diagnosis: SeriesDiagnosis, catalog: list[ForecastMethod]) -> tuple[str, ...]:
    available = {m.name for m in catalog}
    if diagnosis.regime in _REGIME_NAMES:
        wanted = _REGIME_NAMES[diagnosis.regime]
    else:
        wanted = tuple(m.name for m in catalog)
    return tuple(name for name in wanted if name in available)


def filter_catalog(diagnosis: SeriesDiagnosis, catalog: list[ForecastMethod]) -> list[ForecastMethod]:
    names = set(candidate_names(diagnosis, catalog))
    return [m for m in catalog if m.name in names]
