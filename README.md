# Forecasting Agent

**Autonomous demand forecasting with multi-method ensemble, volatility segmentation, and client-facing recommendations.**

An intelligent agent that orchestrates end-to-end time-series forecasting workflows: from product segmentation via coefficient of variation (CV) and changepoints, through Prophet and Prophet+ETS testing, to holdout WAPE and client recommendations.

## Vision

Today's forecasting is manual and siloed:
- ❌ Data analysts test methods in notebooks
- ❌ Accuracy metrics stay hidden in spreadsheets
- ❌ Client decisions are based on gut feel, not data
- ❌ Each new product category requires re-tuning from scratch

**Forecasting Agent** changes this:
- ✅ Automatically segments products by volatility (CV analysis)
- ✅ Diagnoses each SKU (CV, changepoints, zeros, history length, recent level shifts)
- ✅ Routes a candidate set (not always every model), scores holdout WAPE, then picks a method or a CV prior
- ✅ Validates accuracy (WAPE) and flags SKUs for review when error is high
- ✅ Recommends optimal forecast method + horizon per segment
- ✅ Generates client reports with confidence intervals and category groupings
- ✅ Re-forecasts adaptively based on accuracy thresholds

## Key Features

### 1. Volatility Segmentation
Groups products by Coefficient of Variation (CV = STDDEV/AVG):
- **Stable** (CV < 1.0): Conservative Prophet config, shorter forecast horizon
- **Volatile** (CV ≥ 1.0): More flexible Prophet config, ensemble weights lean toward ETS

### 2. Multi-Method Testing
Catalog:
- **Prophet**
- **Baseline** (30-day moving average)
- **Ensemble MA** (Prophet + baseline)
- **ETS** (Holt-Winters)
- **Ensemble ETS** (Prophet + ETS)

The agent **diagnoses** the series first (short history, intermittent zeros, structural break, or regular stable/volatile). Hard regimes get a small robust set; regular series get the full catalog. CV and changepoints **tune** Prophet/ensemble knobs and act as a **prior** when holdout WAPEs are close, missing, or too high (escalate for review).

### 3. Accuracy Validation
Holdout-based backtesting:
- Train on historical data
- Forecast holdout period
- Compare to actuals → compute WAPE
- Track which method wins per segment

### 4. Safety stock from forecast error
After a method is chosen, the agent **backfills** rolling-origin forecasts vs actuals over lead time (TAOS 21 days, DAWBU 42). Safety stock is \(z \sigma_e + \max(0, \text{bias})\), not raw demand CV. It skips auto-SS (escalate) for intermittent, structural break, short history, or high-WAPE reviews. JSON also includes 90/95/99% service-level units.

### 5. Client Recommendations
Actionable insights:
- "Segment A (15 SKUs) should use **Ensemble** (42% WAPE)"
- "Segment B (8 SKUs) wins with **Prophet** (28% WAPE)"
- "Optimal forecast horizons: 60 days (TAOS), 90 days (DAWBU)"
- "These 23 SKUs cluster together — use shared Prophet config"

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

python -m forecasting_agent --input examples/sample_taos_input.csv --brand TAOS
pytest
```

## Architecture

**Agent Workflow:**

```
1. INGEST       → Load 2 years historical sales data
   ↓
2. SEGMENT      → Compute CV per SKU → Stable | Volatile
   ↓
3. TEST         → Fit Prophet and Prophet+ETS (CV + changepoints tune)
   ↓
4. VALIDATE     → Holdout backtest → WAPE scores
   ↓
5. RECOMMEND    → Pick method (WAPE or CV prior)
   ↓
6. SAFETY STOCK → Rolling-origin forecast error → z·σ (+ bias)
   ↓
7. REPORT       → JSON summary (accuracy, decisions, safety stock)
```

**Data Flow:**

```
BigQuery (raw sales) 
    ↓
Agent Orchestrator (LangGraph)
    ├─→ Segmentation Workflow (CV analysis)
    ├─→ Forecasting Workflow (Prophet / Prophet+ETS)
    ├─→ Validation Workflow (WAPE, accuracy tracking)
    └─→ Recommendations Workflow (client insights)
    ↓
BigQuery (forecast output)
    ↓
Client Report (PDF/HTML)
```

## Use Cases

### Case 1: Multi-Brand Forecasting
Same agent runs TAOS + DAWBU automatically:
- Detects brand-specific volatility patterns
- Recommends different methods per brand
- Stores results in separate tables
- Unified reporting view

### Case 2: Product Category Strategy
Client: "Should we forecast Plant A and Plant B together?"

Agent Analysis:
- Plant A: CV=1.2 (volatile) → Ensemble recommended
- Plant B: CV=1.1 (volatile) → Ensemble recommended
- Historical correlation: 0.78
- **Recommendation**: Yes, group together. Use Ensemble with shared horizon=90 days.

### Case 3: New Product Launch
New SKU arrives with no history:
- Agent assigns to "closest cluster" (by category + velocity)
- Uses that cluster's tuned Prophet config
- Flags for re-validation once 6 months history available

### Case 4: Accuracy Troubleshooting
Forecast accuracy drops on Product X:
- Agent identifies root cause (seasonality shift? supply gap?)
- Tests alternative Prophet configs + ensemble weights
- Suggests data quality fixes (too many zero days?)
- Escalates if human review needed

## Repository Structure

```
forecasting-agent/
├── pyproject.toml
├── src/forecasting_agent/
│   ├── domain/           # SalesSeries, results (no I/O)
│   ├── metrics/          # CV, WAPE, changepoints
│   ├── config/           # brand / segment knobs
│   ├── connectors/       # SalesStore: CSV, memory, BigQuery stub
│   ├── methods/          # ForecastMethod: prophet, ets, ensemble
│   ├── segmentation/
│   ├── forecasting/      # parallel/sync runner over methods
│   ├── inventory/        # rolling-origin safety stock from forecast error
│   ├── validation/
│   ├── recommendations/
│   ├── reporting/
│   ├── orchestration/    # ForecastPipeline + optional LangGraph
│   └── cli.py
├── tests/
├── examples/
└── .github/workflows/test.yml
```

## Integration

- **BigQuery**: Source data, forecast storage, accuracy tracking
- **GCP Cloud Functions**: Serverless Prophet execution (python311)
- **LangGraph**: Agent orchestration & decision nodes
- **Pandas/Prophet**: Forecasting engine
- **GitHub Actions**: Weekly execution + notifications

## Status

**Phase**: Phase 1 implementation (Aug 2026)
- [x] Vision & requirements defined
- [x] Architecture sketched
- [x] Phase 1: Core implementation (CV + multi-method testing)
- [x] Phase 2 core: holdout WAPE + accuracy JSON
- [ ] Phase 3: Client HTML/PDF reports & re-forecasting
- [ ] Phase 4: Deployment & monitoring

## Next Steps

1. Implement Phase 1 (CV segmentation + Prophet/Baseline/Ensemble)
2. Write test suite for accuracy validation
3. Build client report generator
4. Deploy agent to production schedule

See [PLAN.md](PLAN.md) for detailed implementation roadmap.

---

**Owner**: Amit Mohanty  
**Repository**: yavdaanalytics/forecasting-agent (private)  
**License**: Private
