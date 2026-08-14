from datetime import date

import numpy as np
import pytest

from forecasting_agent.domain.types import SalesSeries
from forecasting_agent.metrics.cv import classify_cv, coefficient_of_variation


def test_cv_stable_low_noise():
    values = np.array([10.0, 11.0, 9.0, 10.0, 10.5])
    cv, mean, _ = coefficient_of_variation(values)
    assert mean == pytest.approx(10.1)
    assert cv is not None and cv < 1.0
    assignment = classify_cv("S", values)
    assert assignment.segment == "stable"


def test_cv_volatile():
    values = np.array([0.0, 0.0, 0.0, 30.0, 0.0, 30.0, 0.0, 1.0])
    assignment = classify_cv("V", values)
    assert assignment.segment == "volatile"
    assert assignment.cv is not None and assignment.cv >= 1.0


def test_zero_sales_insufficient():
    values = np.zeros(30)
    assignment = classify_cv("Z", values)
    assert assignment.segment == "insufficient"
    assert assignment.reason == "zero_mean"


def test_single_point_insufficient():
    assignment = classify_cv("ONE", np.array([5.0]))
    assert assignment.segment == "insufficient"
    assert assignment.reason == "too_few_points"


def test_outlier_inflates_cv():
    stable = np.full(40, 10.0)
    spiked = stable.copy()
    spiked[-1] = 200.0
    a = classify_cv("O", spiked)
    b = classify_cv("B", stable)
    assert a.cv is not None and b.cv is not None
    assert a.cv > b.cv


def test_sales_series_length_mismatch():
    with pytest.raises(ValueError):
        SalesSeries(sku="X", dates=(date(2024, 1, 1),), values=(1.0, 2.0))


def test_tune_segment_config_regimes():
    from forecasting_agent.config.settings import get_brand_config, tune_segment_config

    taos = get_brand_config("TAOS")
    stable_few = tune_segment_config(taos, "stable", 0.5)
    assert stable_few.changepoint_prior_scale == 0.05
    assert stable_few.prophet_weight == 0.7
    assert stable_few.horizon == 60

    stable_many = tune_segment_config(taos, "stable", 2.0)
    assert stable_many.changepoint_prior_scale == 0.15
    assert stable_many.prophet_weight == 0.5

    vol_few = tune_segment_config(taos, "volatile", 1.0)
    assert vol_few.changepoint_prior_scale == 0.15
    assert vol_few.prophet_weight == 0.5

    vol_many = tune_segment_config(taos, "volatile", 3.0)
    assert vol_many.changepoint_prior_scale == 0.5
    assert vol_many.prophet_weight == 0.3

    dawbu = tune_segment_config("DAWBU", "volatile", 4.0)
    assert dawbu.changepoint_prior_scale == 0.3
    assert dawbu.horizon == 90
