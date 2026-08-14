"""Forecasting agent: CV segmentation, multi-method forecasts, validation."""

from forecasting_agent.config.settings import BrandConfig, get_brand_config
from forecasting_agent.domain.types import ForecastResult, SalesSeries, SegmentAssignment
from forecasting_agent.orchestration.pipeline import ForecastPipeline

__all__ = [
    "BrandConfig",
    "ForecastPipeline",
    "ForecastResult",
    "SalesSeries",
    "SegmentAssignment",
    "get_brand_config",
]

__version__ = "0.1.0"
