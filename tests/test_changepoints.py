import numpy as np

from forecasting_agent.metrics.changepoints import detect_changepoints


def test_flat_series_no_breaks():
    values = np.full(90, 10.0)
    n_cp, per_year = detect_changepoints(values)
    assert n_cp == 0
    assert per_year == 0.0


def test_level_shift_detects_break():
    values = np.concatenate([np.full(40, 5.0), np.full(40, 25.0)])
    n_cp, per_year = detect_changepoints(values)
    assert n_cp >= 1
    assert per_year > 0
