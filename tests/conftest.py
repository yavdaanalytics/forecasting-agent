import numpy as np
import pytest

from forecasting_agent.domain.types import SalesSeries
from tests.helpers import make_series


@pytest.fixture
def stable_series() -> SalesSeries:
    rng = np.random.default_rng(0)
    values = (10 + rng.normal(0, 0.8, 120)).clip(min=0).tolist()
    return make_series("PLANT-001", values, category="plants")


@pytest.fixture
def volatile_series() -> SalesSeries:
    rng = np.random.default_rng(1)
    base = rng.choice([0.0, 0.0, 0.0, 8.0, 20.0], size=120)
    return make_series("PLANT-301", base.tolist(), category="plants")
