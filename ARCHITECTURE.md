# Architecture: Forecasting Agent

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Forecasting Agent                              │
│                    (LangGraph Orchestrator)                           │
└──────────────────────────────────────────────────────────────────────┘
         │
         ├─→ Segmentation Engine      (CV Analysis)
         ├─→ Forecasting Engine       (Prophet/Baseline/Ensemble)
         ├─→ Validation Engine        (WAPE, Backtesting)
         ├─→ Recommendation Engine    (Client Insights)
         └─→ Reporting Engine         (Client Reports)
         │
    ┌────┴─────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
BigQuery                            GCP Cloud Functions
(Forecast Storage)                  (Python 3.11 Remote)
```

## Key Components

### 1. Agent Orchestrator (LangGraph)
Manages workflow state machine with nodes:
- **segmentation**: CV analysis per SKU
- **forecasting**: Prophet/Baseline/Ensemble (parallel)
- **validation**: Holdout backtesting → WAPE per method
- **recommendations**: Best method + horizon per segment
- **reporting**: Client-facing document generation

### 2. Segmentation Workflow
Classifies products by volatility (CV = STDDEV/AVG):
- **Stable** (CV < 1.0): Conservative config, 60-day horizon
- **Volatile** (CV ≥ 1.0): Flexible config, 90-day horizon, ensemble preferred

### 3. Forecasting Workflow
Tests three methods per segment (parallel execution):
- **Prophet**: Time-series with trend & seasonality (changepoint-sensitive)
- **Baseline**: Simple moving average
- **Ensemble**: 50% Prophet + 50% Baseline (hedges overfitting)

Integration with GCP Cloud Functions (forecast_fn):
- Calls BigQuery Remote Function with explicit parameters
- Batch processing: 50 SKUs per batch
- Response time: 30-90 seconds per batch

### 4. Validation Workflow
Holdout-based accuracy testing:
- Hold out 30 days from end of history
- Run Prophet on remaining 80% of data
- Forecast into holdout period
- Compare yhat vs actual → compute WAPE
- Identify best method per segment

### 5. Recommendations Workflow
Generates actionable insights:
- "Segment A (15 SKUs) → Ensemble (42% WAPE)"
- "Segment B (8 SKUs) → Prophet (28% WAPE)"
- "Suggest grouping these 23 SKUs together by category"

### 6. Reporting Workflow
Client-facing document with:
- Executive summary (methods tested, accuracy results)
- Per-segment recommendations with confidence levels
- Actionable next steps
- Appendix: detailed accuracy metrics per SKU

## Data Flow

```
BigQuery (Raw: 2 years history)
    ↓
Segmentation (CV analysis) → stable/volatile groups
    ↓
Forecasting (Prophet/Baseline/Ensemble per group)
    ↓
Validation (Holdout backtest → WAPE per method)
    ↓
Recommendations (Best method per segment)
    ↓
Reporting (HTML/PDF client document)
    ↓
BigQuery (Store: forecasts, accuracy, recommendations)
```

## Integration Points

- **BigQuery**: Data source & output storage
- **GCP Cloud Functions**: forecast_fn (Python 3.11, serverless)
- **LangGraph**: Workflow orchestration & state management
- **Pandas/Prophet**: Core forecasting library
- **GitHub Actions**: Weekly scheduling + notifications

## Configuration

Per-brand, per-segment Prophet tuning:

```python
BRAND_CONFIG = {
    "TAOS": {
        "stable": {"changepoint_prior_scale": 0.05, "horizon": 60},
        "volatile": {"changepoint_prior_scale": 0.5, "horizon": 60}
    },
    "DAWBU": {
        "stable": {"changepoint_prior_scale": 0.05, "horizon": 90},
        "volatile": {"changepoint_prior_scale": 0.3, "horizon": 90}
    }
}
```

Tuning methodology: Offline cross-validation on historical data per brand.

## Cost Breakdown

**Weekly Execution (1 call/week):**
- GCP Cloud Functions: $0 (within 400K GB-sec free tier)
- BigQuery: ~$5-10/week (storage + queries)
- **Total: ~$0-10/week**

Even at 10x scale (10 calls/week), still within free tier.

## Monitoring

Track per execution:
- # SKUs per segment
- Execution time (segmentation → forecasting → validation → reporting)
- WAPE per method per segment
- Report generation time
- Recommendation adoption rate (how many clients use?)

Metrics stored in BigQuery for trend analysis.

## Deployment Targets

1. **Local Development**: `python -m src.agent.run_forecast_pipeline`
2. **GitHub Actions**: Weekly cron job + notifications
3. **Cloud Scheduler** (optional): GCP managed scheduler

## Future: Multi-Brand Scaling

- Auto-discover optimal horizon per brand
- Brand clustering (which forecast similarly?)
- Federated learning (TAOS insights → DAWBU)
