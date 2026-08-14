from __future__ import annotations

from forecasting_agent.methods.baseline import MovingAverageBaseline
from forecasting_agent.methods.ensemble import EnsembleMethod
from forecasting_agent.methods.ets import ETSMethod
from forecasting_agent.methods.prophet import ProphetMethod, ProphetUnavailableError, _import_prophet
from forecasting_agent.methods.protocol import ForecastMethod


def prophet_is_available() -> bool:
    try:
        _import_prophet()
    except ProphetUnavailableError:
        return False
    return True


def build_methods(*, baseline_window: int = 30, require_prophet: bool = False) -> list[ForecastMethod]:
    """Full catalog: Prophet, MA baseline, Prophet+MA, ETS, Prophet+ETS.

    If Prophet is missing, drop Prophet-backed methods unless require_prophet.
    """
    baseline = MovingAverageBaseline(baseline_window)
    ets = ETSMethod()
    if prophet_is_available():
        prophet = ProphetMethod()
        return [
            prophet,
            baseline,
            EnsembleMethod(prophet, baseline, name="ensemble_ma"),
            ets,
            EnsembleMethod(prophet, ets, name="ensemble_ets"),
        ]
    if require_prophet:
        raise ProphetUnavailableError("prophet is required for this run")
    return [
        baseline,
        ets,
        EnsembleMethod(ets, baseline, name="ensemble_ma"),
    ]
