from forecasting_agent.config.settings import STABLE_DEFAULT, get_brand_config
from forecasting_agent.domain.policy import MethodDecision, SeriesDiagnosis
from forecasting_agent.inventory.backfill import lead_time_errors
from forecasting_agent.inventory.compute import compute_safety_stock
from forecasting_agent.inventory.formula import safety_stock_units, z_for_service_level
from forecasting_agent.methods.baseline import MovingAverageBaseline
from tests.helpers import make_series


def _diag(sku: str, regime: str = "stable") -> SeriesDiagnosis:
    return SeriesDiagnosis(
        sku=sku,
        regime=regime,  # type: ignore[arg-type]
        n_points=120,
        zero_fraction=0.05,
        recent_shift_ratio=1.0,
        cv_segment="stable",
        changepoints_per_year=0.2,
        reason="cv_stable",
    )


def _decision(sku: str, method: str = "baseline", *, escalate: bool = False) -> MethodDecision:
    return MethodDecision(
        sku=sku,
        method=method,
        reason="holdout_wape",
        confidence="high",
        escalate=escalate,
        wape=0.1,
        candidates=("baseline",),
    )


def test_z_for_known_service_levels():
    assert abs(z_for_service_level(0.95) - 1.64485) < 1e-4
    assert abs(z_for_service_level(0.90) - 1.28155) < 1e-4


def test_ss_covers_underforecast_bias():
    from_std_only = safety_stock_units(10.0, service_level=0.95, error_bias=0.0)
    with_bias = safety_stock_units(10.0, service_level=0.95, error_bias=5.0)
    assert with_bias == from_std_only + 5.0
    assert safety_stock_units(10.0, service_level=0.95, error_bias=-8.0) == from_std_only


def test_backfill_constant_series_near_zero_error():
    series = make_series("C", [10.0] * 120)
    errors = lead_time_errors(
        series,
        MovingAverageBaseline(window=30),
        lead_time_days=7,
        stride=7,
        max_origins=5,
        min_train=40,
    )
    assert len(errors) >= 3
    assert all(abs(e) < 1e-6 for e in errors)


def test_skip_intermittent():
    series = make_series("I", [10.0] * 120)
    result = compute_safety_stock(
        series,
        _diag("I", "intermittent"),
        _decision("I"),
        MovingAverageBaseline(),
        get_brand_config("TAOS"),
        STABLE_DEFAULT,
    )
    assert result.escalate is True
    assert result.safety_stock is None
    assert result.reason == "skip_intermittent"


def test_skip_escalated_forecast():
    series = make_series("E", [10.0] * 120)
    result = compute_safety_stock(
        series,
        _diag("E"),
        _decision("E", escalate=True),
        MovingAverageBaseline(),
        get_brand_config("TAOS"),
        STABLE_DEFAULT,
    )
    assert result.reason == "skip_escalated_forecast"
    assert result.escalate is True


def test_compute_ss_on_stable_baseline():
    series = make_series("S", [10.0] * 120)
    result = compute_safety_stock(
        series,
        _diag("S"),
        _decision("S"),
        MovingAverageBaseline(),
        get_brand_config("TAOS"),
        STABLE_DEFAULT,
    )
    assert result.escalate is False
    assert result.n_origins >= 3
    assert result.safety_stock is not None
    assert result.safety_stock >= 0
    assert "0.95" in result.by_service_level
