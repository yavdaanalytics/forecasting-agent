from forecasting_agent.domain.policy import SeriesDiagnosis
from forecasting_agent.methods.baseline import MovingAverageBaseline
from forecasting_agent.methods.ets import ETSMethod
from forecasting_agent.policy.candidates import candidate_names
from forecasting_agent.policy.diagnose import diagnose
from forecasting_agent.policy.select import select_method
from forecasting_agent.segmentation.classify import assign_segments
from tests.helpers import make_series


def _diag(**overrides) -> SeriesDiagnosis:
    base = dict(
        sku="X",
        regime="stable",
        n_points=120,
        zero_fraction=0.1,
        recent_shift_ratio=1.0,
        cv_segment="stable",
        changepoints_per_year=0.5,
        reason="cv_stable",
    )
    base.update(overrides)
    return SeriesDiagnosis(**base)


def test_short_history_regime():
    series = make_series("S", [10.0] * 40)
    assignment = assign_segments([series])[0]
    diag = diagnose(series, assignment)
    assert diag.regime == "short_history"


def test_intermittent_regime():
    values = [0.0] * 80 + [5.0] * 20
    series = make_series("I", values)
    assignment = assign_segments([series])[0]
    diag = diagnose(series, assignment)
    assert diag.regime == "intermittent"
    assert diag.zero_fraction >= 0.6


def test_structural_break_regime():
    values = [5.0] * 90 + [20.0] * 30
    series = make_series("B", values)
    assignment = assign_segments([series])[0]
    diag = diagnose(series, assignment)
    assert diag.regime == "structural_break"


def test_candidates_short_history():
    catalog = [MovingAverageBaseline(), ETSMethod()]
    names = candidate_names(_diag(regime="short_history"), catalog)
    assert names == ("baseline", "ets")


def test_select_clear_wape_winner():
    decision = select_method(
        _diag(),
        {"prophet": 0.20, "ensemble_ets": 0.40},
        ("prophet", "ensemble_ets"),
    )
    assert decision.method == "prophet"
    assert decision.reason == "holdout_wape"
    assert decision.escalate is False


def test_select_close_wape_uses_cv_prior():
    decision = select_method(
        _diag(cv_segment="stable", changepoints_per_year=0.4, regime="stable"),
        {"prophet": 0.28, "ensemble_ets": 0.30},
        ("prophet", "ensemble_ets"),
    )
    assert decision.method == "prophet"
    assert decision.reason == "cv_prior_close_wape"


def test_select_high_wape_escalates():
    decision = select_method(
        _diag(),
        {"prophet": 0.62, "baseline": 0.70},
        ("prophet", "baseline"),
    )
    assert decision.escalate is True
    assert decision.reason == "high_wape_review"


def test_select_missing_holdout_uses_prior():
    decision = select_method(
        _diag(regime="intermittent", cv_segment="volatile"),
        {"baseline": None, "ets": None},
        ("baseline", "ets"),
    )
    assert decision.method == "baseline"
    assert decision.reason == "cv_prior_no_holdout"
    assert decision.escalate is True
