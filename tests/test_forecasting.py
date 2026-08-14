from datetime import timedelta

from forecasting_agent.domain.types import ForecastResult
from forecasting_agent.methods.baseline import MovingAverageBaseline
from forecasting_agent.methods.ensemble import EnsembleMethod
from forecasting_agent.methods.ets import ETSMethod
from tests.helpers import make_series


class _StubMethod:
    def __init__(self, name: str, level: float) -> None:
        self.name = name
        self.level = level

    def fit_predict(self, series, *, horizon, config=None):
        start = series.dates[-1] + timedelta(days=1)
        dates = tuple(start + timedelta(days=i) for i in range(horizon))
        return ForecastResult(
            sku=series.sku,
            method=self.name,
            dates=dates,
            values=tuple(self.level for _ in range(horizon)),
        )


def test_baseline_shape_and_level():
    series = make_series("B", [10.0] * 40)
    result = MovingAverageBaseline(window=30).fit_predict(series, horizon=7)
    assert result.method == "baseline"
    assert len(result.values) == 7
    assert all(v == 10.0 for v in result.values)
    assert result.dates[0] > series.dates[-1]


def test_ensemble_blends_prophet_and_ets_stubs():
    series = make_series("E", list(range(1, 31)))
    prophet = _StubMethod("prophet", 10.0)
    ets = _StubMethod("ets", 4.0)
    ensemble = EnsembleMethod(prophet, ets, primary_weight=0.7)
    result = ensemble.fit_predict(series, horizon=5)
    assert result.values[0] == 0.7 * 10.0 + 0.3 * 4.0
    assert result.method == "ensemble"
    assert result.metadata["primary"] == "prophet"
    assert result.metadata["secondary"] == "ets"


def test_ets_horizon_and_non_negative():
    series = make_series("ETS", [8.0] * 28)
    result = ETSMethod().fit_predict(series, horizon=7)
    assert result.method == "ets"
    assert len(result.values) == 7
    assert all(v >= 0.0 for v in result.values)
    assert result.dates[0] > series.dates[-1]


def test_empty_series_baseline():
    from forecasting_agent.domain.types import SalesSeries

    empty = SalesSeries(sku="X", dates=(), values=())
    result = MovingAverageBaseline().fit_predict(empty, horizon=3)
    assert result.values == (0.0, 0.0, 0.0)
