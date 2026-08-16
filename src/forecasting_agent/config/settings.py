from __future__ import annotations

from dataclasses import dataclass, replace

CV_STABLE_THRESHOLD = 1.0
DEFAULT_HOLDOUT_DAYS = 30
DEFAULT_BASELINE_WINDOW = 30
DEFAULT_ENSEMBLE_PROPHET_WEIGHT = 0.5


@dataclass(frozen=True)
class SegmentConfig:
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    weekly_seasonality: bool
    yearly_seasonality: bool
    daily_seasonality: bool
    horizon: int
    prophet_weight: float


STABLE_DEFAULT = SegmentConfig(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=False,
    horizon=60,
    prophet_weight=0.7,
)

VOLATILE_DEFAULT = SegmentConfig(
    changepoint_prior_scale=0.3,
    seasonality_prior_scale=3.0,
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=False,
    horizon=90,
    prophet_weight=0.3,
)


@dataclass(frozen=True)
class BrandConfig:
    name: str
    stable: SegmentConfig
    volatile: SegmentConfig
    holdout_days: int = DEFAULT_HOLDOUT_DAYS
    baseline_window: int = DEFAULT_BASELINE_WINDOW
    lead_time_days: int = 21
    review_period_days: int = 7
    service_level: float = 0.95
    backfill_stride: int = 7
    backfill_max_origins: int = 6


_BRANDS: dict[str, BrandConfig] = {
    "TAOS": BrandConfig(
        name="TAOS",
        stable=replace(STABLE_DEFAULT, horizon=60),
        volatile=replace(VOLATILE_DEFAULT, changepoint_prior_scale=0.5, horizon=60),
    ),
    "DAWBU": BrandConfig(
        name="DAWBU",
        stable=replace(STABLE_DEFAULT, horizon=90),
        volatile=replace(VOLATILE_DEFAULT, changepoint_prior_scale=0.3, horizon=90),
        lead_time_days=42,
    ),
}


def get_brand_config(name: str) -> BrandConfig:
    key = name.strip().upper()
    if key not in _BRANDS:
        known = ", ".join(sorted(_BRANDS))
        raise KeyError(f"Unknown brand {name!r}. Known: {known}")
    return _BRANDS[key]


def select_horizon(cv: float, avg_sales_per_day: float, brand: BrandConfig | None = None) -> int:
    """Horizon from volatility and velocity; brand config wins when provided."""
    if brand is not None:
        if cv < CV_STABLE_THRESHOLD:
            return brand.stable.horizon
        return brand.volatile.horizon
    if cv < 1.0:
        return 60
    if avg_sales_per_day < 2:
        return 90
    return 60


def segment_config_for(brand: BrandConfig, segment: str) -> SegmentConfig:
    if segment == "stable":
        return brand.stable
    if segment == "volatile":
        return brand.volatile
    return brand.volatile


CHANGEPOINTS_FEW_PER_YEAR = 2.0


def tune_segment_config(
    brand: BrandConfig | str,
    cv_segment: str,
    changepoints_per_year: float,
) -> SegmentConfig:
    """Per-SKU Prophet flexibility and ensemble weight from CV + changepoints/year."""
    if isinstance(brand, str):
        brand = get_brand_config(brand)
    few = changepoints_per_year < CHANGEPOINTS_FEW_PER_YEAR
    if cv_segment == "stable":
        base = brand.stable
        if few:
            return replace(base, changepoint_prior_scale=0.05, prophet_weight=0.7)
        return replace(base, changepoint_prior_scale=0.15, prophet_weight=0.5)
    base = brand.volatile
    if few:
        return replace(base, changepoint_prior_scale=0.15, prophet_weight=0.5)
    return replace(base, prophet_weight=0.3)
