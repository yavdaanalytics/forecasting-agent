import numpy as np
import pytest

from forecasting_agent.metrics.wape import wape
from forecasting_agent.validation.holdout import split_holdout
from tests.helpers import make_series


def test_wape_known_values():
    actual = np.array([10.0, 10.0, 10.0])
    pred = np.array([8.0, 12.0, 10.0])
    assert wape(actual, pred) == pytest.approx(4.0 / 30.0)


def test_wape_zero_actuals():
    assert wape(np.zeros(5), np.ones(5)) is None


def test_wape_empty():
    assert wape(np.array([]), np.array([])) is None


def test_holdout_split():
    series = make_series("H", list(range(100)))
    train, holdout = split_holdout(series, 20)
    assert len(holdout.dates) == 20
    assert len(train.dates) == 80
    assert train.dates[-1] < holdout.dates[0]


def test_holdout_too_short():
    series = make_series("H", [1.0, 2.0, 3.0])
    try:
        split_holdout(series, 10)
        assert False, "expected ValueError"
    except ValueError:
        pass
