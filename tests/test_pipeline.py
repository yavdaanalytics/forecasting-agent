from forecasting_agent.connectors.csv_store import MemorySalesStore
from forecasting_agent.methods.factory import build_methods, prophet_is_available
from forecasting_agent.orchestration.pipeline import ForecastPipeline
from forecasting_agent.reporting.accuracy import pipeline_as_dict
from tests.helpers import make_series


def test_build_methods_catalog():
    names = [m.name for m in build_methods()]
    assert "baseline" in names
    assert "ets" in names
    assert "ensemble_ma" in names
    if prophet_is_available():
        assert names == ["prophet", "baseline", "ensemble_ma", "ets", "ensemble_ets"]
    else:
        assert "prophet" not in names
        assert "ensemble_ets" not in names


def test_pipeline_segments_and_forecasts(stable_series, volatile_series):
    store = MemorySalesStore([stable_series, volatile_series])
    pipeline = ForecastPipeline(store, brand="TAOS", methods=build_methods())
    result = pipeline.run()
    segments = {a.sku: a.segment for a in result.assignments}
    assert segments["PLANT-001"] == "stable"
    assert segments["PLANT-301"] == "volatile"
    assert result.segments["stable"].count == 1
    assert result.forecasts
    assert result.decisions
    payload = pipeline_as_dict(result)
    assert payload["brand"] == "TAOS"
    assert "stable" in payload["recommendations"]
    rec_method = payload["recommendations"]["stable"]["method"]
    assert rec_method in {"prophet", "baseline", "ensemble_ma", "ets", "ensemble_ets"}
    stable_diag = result.diagnoses["PLANT-001"]
    assert stable_diag.regime in {"stable", "structural_break"}
    vol_diag = result.diagnoses["PLANT-301"]
    assert vol_diag.regime in {"volatile", "intermittent", "structural_break"}


def test_pipeline_skips_insufficient():
    dead = make_series("DEAD", [0.0] * 80)
    store = MemorySalesStore([dead])
    result = ForecastPipeline(store, brand="TAOS").run()
    assert result.assignments[0].segment == "insufficient"
    assert result.forecasts == ()
    assert result.decisions[0].escalate is True


def test_short_history_only_cheap_methods():
    series = make_series("SHORT", [8.0] * 40)
    result = ForecastPipeline(MemorySalesStore([series]), brand="TAOS").run()
    used = {f.method for f in result.forecasts}
    assert used <= {"baseline", "ets"}
    assert result.diagnoses["SHORT"].regime == "short_history"
