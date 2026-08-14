# Forecasting Agent

**Autonomous demand forecasting with multi-method ensemble, volatility segmentation, and client-facing recommendations.**

An intelligent agent that orchestrates end-to-end time-series forecasting workflows: from product segmentation via coefficient of variation (CV), through multi-method forecast testing (Prophet, Baseline, Ensemble), to accuracy validation and actionable client recommendations.

## Vision

Today's forecasting is manual and siloed:
- ❌ Data analysts test methods in notebooks
- ❌ Accuracy metrics stay hidden in spreadsheets
- ❌ Client decisions are based on gut feel, not data
- ❌ Each new product category requires re-tuning from scratch

**Forecasting Agent** changes this:
- ✅ Automatically segments products by volatility (CV analysis)
- ✅ Tests Prophet, Baseline, Ensemble in parallel for each segment
- ✅ Validates accuracy (WAPE) and flags underperformers
- ✅ Recommends optimal forecast method + horizon per segment
- ✅ Generates client reports with confidence intervals and category groupings
- ✅ Re-forecasts adaptively based on accuracy thresholds

## Key Features

### 1. Volatility Segmentation
Groups products by Coefficient of Variation (CV = STDDEV/AVG):
- **Stable** (CV < 1.0): Conservative Prophet config, shorter forecast horizon
- **Volatile** (CV ≥ 1.0): Flexible Prophet config, ensemble methods preferred

### 2. Multi-Method Testing
Tests in parallel for each segment:
- **Prophet** (time-series with changepoint detection)
- **Baseline** (simple moving average)
- **Ensemble** (weighted blend: 50% Prophet + 50% Baseline)

### 3. Accuracy Validation
Holdout-based backtesting:
- Train on historical data
- Forecast holdout period
- Compare to actuals → compute WAPE
- Track which method wins per segment

### 4. Client Recommendations
Actionable insights:
- "Segment A (15 SKUs) should use **Ensemble** (42% WAPE)"
- "Segment B (8 SKUs) wins with **Prophet** (28% WAPE)"
- "Optimal forecast horizons: 60 days (TAOS), 90 days (DAWBU)"
- "These 23 SKUs cluster together — use shared Prophet config"

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run end-to-end pipeline
python src/agent/run_forecast_pipeline.py --brand TAOS --mode full

# Generate client report
python src/reporting/generate_client_report.py --brand DAWBU
```

## Architecture

**Agent Workflow:**

```
1. INGEST       → Load 2 years historical sales data
   ↓
2. SEGMENT      → Compute CV per SKU → Stable | Volatile
   ↓
3. TEST         → Run Prophet/Baseline/Ensemble per segment (parallel)
   ↓
4. VALIDATE     → Holdout backtest → WAPE scores
   ↓
5. RECOMMEND    → Pick best method per segment + suggest groupings
   ↓
6. DEPLOY       → Run production forecast with tuned config
   ↓
7. REPORT       → Generate client-facing document
```

**Data Flow:**

```
BigQuery (raw sales) 
    ↓
Agent Orchestrator (LangGraph)
    ├─→ Segmentation Workflow (CV analysis)
    ├─→ Forecasting Workflow (Prophet/Baseline/Ensemble)
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
├── README.md                    # Overview & quick start
├── ARCHITECTURE.md              # Technical design & components
├── PLAN.md                      # Implementation roadmap
├── IDEAS.md                     # Research & conceptual depth
├── requirements.txt             # Dependencies
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # LangGraph agent
│   │   ├── run_forecast_pipeline.py
│   │   └── workflows/
│   │       ├── segmentation.py      # CV analysis
│   │       ├── forecasting.py       # Prophet/Baseline/Ensemble
│   │       ├── validation.py        # WAPE, backtesting
│   │       └── recommendations.py   # Client insights
│   ├── models/
│   │   ├── prophet_wrapper.py
│   │   ├── baseline_wrapper.py
│   │   └── ensemble_wrapper.py
│   ├── connectors/
│   │   └── bigquery.py          # BQ integration
│   └── reporting/
│       ├── accuracy_report.py
│       └── client_recommendations.py
├── tests/
│   ├── test_segmentation.py
│   ├── test_forecasting.py
│   └── test_validation.py
├── examples/
│   ├── sample_taos_input.csv
│   ├── sample_output.json
│   └── sample_client_report.md
└── .github/
    └── workflows/
        └── forecast_weekly.yml  # GitHub Actions
```

## Integration

- **BigQuery**: Source data, forecast storage, accuracy tracking
- **GCP Cloud Functions**: Serverless Prophet execution (python311)
- **LangGraph**: Agent orchestration & decision nodes
- **Pandas/Prophet**: Forecasting engine
- **GitHub Actions**: Weekly execution + notifications

## Status

**Phase**: MVP Planning (Aug 2026)
- [x] Vision & requirements defined
- [x] Architecture sketched
- [ ] Phase 1: Core implementation (CV + multi-method testing)
- [ ] Phase 2: Validation & accuracy reporting
- [ ] Phase 3: Client recommendations & re-forecasting
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
