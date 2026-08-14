# Implementation Plan: Forecasting Agent

**Timeline**: 4-6 weeks | **Status**: Planning (Aug 2026)

## Phase 1: MVP Core (Weeks 1-2)

**Objective**: Build segmentation + multi-method testing framework

### Task 1.1: Project Setup & Scaffolding
- [ ] Initialize repo structure (src/, tests/, examples/)
- [ ] Setup dependencies (pandas, prophet, langgraph, google-cloud-bigquery)
- [ ] Create CI/CD pipeline (GitHub Actions, pytest config)
- [ ] Document setup process in CONTRIBUTING.md

**Deliverable**: Working dev environment, passing empty tests

### Task 1.2: Segmentation Workflow
- [ ] Implement CV analysis per SKU
- [ ] Classify Stable (CV < 1.0) vs Volatile (CV ≥ 1.0)
- [ ] Unit tests for edge cases (zero sales, single data point, outliers)
- [ ] Integration test with sample TAOS/DAWBU data

**Deliverable**: `src/agent/workflows/segmentation.py` with 90%+ test coverage

**Sample Output**:
```json
{
  "stable": {
    "skus": ["PLANT-001", "PLANT-002", ...],
    "count": 142,
    "avg_cv": 0.65,
    "config": {"changepoint_prior_scale": 0.05, "horizon": 60}
  },
  "volatile": {
    "skus": ["PLANT-301", "PLANT-302", ...],
    "count": 73,
    "avg_cv": 1.45,
    "config": {"changepoint_prior_scale": 0.3, "horizon": 90}
  }
}
```

### Task 1.3: Forecasting Workflow (Multi-Method Testing)
- [ ] Implement Prophet wrapper (call forecast_fn with CV-tuned config)
- [ ] Implement Baseline wrapper (simple moving average)
- [ ] Implement Ensemble wrapper (50/50 blend Prophet + Baseline)
- [ ] Async execution (run all 3 methods in parallel per segment)
- [ ] Error handling & retry logic for Cloud Function calls

**Deliverable**: `src/agent/workflows/forecasting.py` with async execution

**Integration Point**: Calls GCP Cloud Functions (forecast_fn) with explicit parameters:
```python
response = forecast_fn(
    dates=dates_json,
    values=values_json,
    horizon=60,
    method="prophet_only",  # or "ensemble", "baseline"
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    ...
)
```

### Task 1.4: Agent Orchestrator (LangGraph)
- [ ] Define ForecastState TypedDict
- [ ] Build StateGraph with nodes: segmentation → forecasting → routing
- [ ] Implement state transitions & error handling
- [ ] Test full pipeline on sample data

**Deliverable**: `src/agent/orchestrator.py` with working state machine

**Test Input**: 100 sample SKUs (50 TAOS, 50 DAWBU) with 2 years history
**Expected Output**: Segmentation + forecasts for each segment

## Phase 2: Validation & Accuracy (Weeks 3-4)

**Objective**: Measure forecast accuracy, identify best method per segment

### Task 2.1: Validation Workflow
- [ ] Implement holdout backtesting (split 80/20)
- [ ] Compute WAPE per method per segment
- [ ] Compare actual vs forecast on holdout period
- [ ] Generate accuracy report (method comparison table)

**Deliverable**: `src/agent/workflows/validation.py`

**Sample Output**:
```json
{
  "stable": {
    "prophet_wape": 0.28,
    "baseline_wape": 0.35,
    "ensemble_wape": 0.32,
    "recommended_method": "prophet"
  },
  "volatile": {
    "prophet_wape": 0.52,
    "baseline_wape": 0.48,
    "ensemble_wape": 0.42,
    "recommended_method": "ensemble"
  }
}
```

### Task 2.2: Accuracy Reporting
- [ ] Generate per-segment accuracy table
- [ ] Rank methods by WAPE
- [ ] Identify outlier SKUs (high error)
- [ ] Suggest data quality issues (too many zeros?)

**Deliverable**: `src/reporting/accuracy_report.py`

### Task 2.3: Integration Tests
- [ ] End-to-end test: raw data → segmentation → forecasting → validation
- [ ] Test with real TAOS/DAWBU data (sample)
- [ ] Verify forecast_fn integration works at scale (50+ SKUs)
- [ ] Performance benchmark (execution time per stage)

**Deliverable**: `tests/test_e2e_pipeline.py` with real data

## Phase 3: Recommendations & Re-Forecasting (Weeks 5-6)

**Objective**: Generate client-facing insights, enable adaptive re-forecasting

### Task 3.1: Recommendation Workflow
- [ ] Analyze per-segment accuracy
- [ ] Recommend method + horizon per segment
- [ ] Suggest SKU groupings (similar volatility, forecast error)
- [ ] Confidence scores (high if WAPE < 40%, medium if 40-50%, low if > 50%)

**Deliverable**: `src/agent/workflows/recommendations.py`

**Sample Output**:
```json
{
  "segments": {
    "stable": {
      "method": "prophet",
      "wape": "28%",
      "horizon_days": 60,
      "confidence": "high",
      "num_skus": 142
    },
    "volatile": {
      "method": "ensemble",
      "wape": "42%",
      "horizon_days": 90,
      "confidence": "medium",
      "num_skus": 73
    }
  },
  "groupings": [
    {
      "name": "Fast-Moving Plants",
      "skus": ["PLANT-001", "PLANT-002", ...],
      "rationale": "Similar CV (1.2-1.4), all volatile, ensemble recommended"
    },
    {
      "name": "Slow-Moving Seeds",
      "skus": ["SEED-001", "SEED-002", ...],
      "rationale": "Stable demand (CV < 0.8), Prophet wins"
    }
  ]
}
```

### Task 3.2: Client Reporting
- [ ] Generate HTML report with visualizations
- [ ] Include executive summary (# SKUs, methods tested, results)
- [ ] Per-segment breakdowns with charts
- [ ] Actionable recommendations section
- [ ] Technical appendix (detailed metrics per SKU)

**Deliverable**: `src/reporting/client_recommendations.py`

**Output Format**: HTML/PDF (Jinja2 templates)

### Task 3.3: Adaptive Re-Forecasting
- [ ] Monitor forecast accuracy over time
- [ ] If accuracy drops > 10%, trigger re-forecasting
- [ ] Test alternative Prophet configs (grid search on changepoint_prior_scale)
- [ ] Re-run ensemble weight tuning
- [ ] Flag for human review if accuracy still low

**Deliverable**: `src/agent/workflows/reforecasting.py`

## Phase 4: Deployment & Monitoring (Week 7+)

**Objective**: Production deployment, metrics tracking, client handoff

### Task 4.1: GitHub Actions Scheduler
- [ ] Create `.github/workflows/forecast_weekly.yml`
- [ ] Schedule: Every Sunday 2 AM UTC
- [ ] Steps: Ingest → Segment → Forecast → Validate → Recommend → Report
- [ ] Notifications: Slack alert on completion (summary + any failures)

**Deliverable**: Automated weekly execution

### Task 4.2: Metrics & Monitoring
- [ ] Track execution time per stage
- [ ] Monitor accuracy trends (WAPE over time)
- [ ] Store metrics in BigQuery (for trend analysis)
- [ ] Create monitoring dashboard (Looker/Data Studio)

**Deliverable**: Observability setup, dashboard

### Task 4.3: Client Handoff
- [ ] Prepare client documentation (how to interpret results)
- [ ] Setup report distribution (email, cloud storage, Power BI link)
- [ ] Create FAQ guide for common questions
- [ ] Train business team on using recommendations

**Deliverable**: Client documentation, report delivery pipeline

## Testing Strategy

### Unit Tests
- Segmentation: CV calculation, edge cases
- Forecasting: Prophet/Baseline/Ensemble output shapes
- Validation: WAPE calculation, holdout split correctness
- Recommendations: Method ranking, confidence scoring

### Integration Tests
- End-to-end: raw data → report
- BigQuery connection: query execution, data loading
- Cloud Functions: forecast_fn integration, batch processing
- Report generation: HTML rendering, PDF export

### Performance Tests
- Segmentation: < 5 seconds for 500 SKUs
- Forecasting: < 120 seconds for 50 SKUs (all 3 methods)
- Validation: < 10 seconds for holdout backtest
- Reporting: < 30 seconds for HTML generation

## Dependencies

**Core**:
- pandas >= 1.3.0
- prophet >= 1.1
- langgraph >= 0.1.0
- google-cloud-bigquery >= 3.0

**Development**:
- pytest >= 7.0
- black, flake8 (code quality)
- sphinx (docs)

**Reporting**:
- jinja2 (HTML templates)
- reportlab (PDF generation)

## Success Criteria

### Phase 1
- [x] Agent orchestrator working
- [x] Segmentation produces correct CV-based classification
- [x] Forecasting calls cloud function successfully
- [x] Tests pass (unit + integration)

### Phase 2
- [x] Validation computes WAPE correctly (vs manual spot-check)
- [x] Best method identified per segment (prophet for stable, ensemble for volatile)
- [x] Accuracy reports generated

### Phase 3
- [x] Recommendations clear & actionable
- [x] Client report readable & visually appealing
- [x] SKU groupings make sense (validated with business team)

### Phase 4
- [x] Automated weekly execution (no manual steps)
- [x] Metrics tracked in BigQuery
- [x] Client receives weekly report with insights
- [x] Zero operational overhead after launch

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Prophet training fails on sparse data | Medium | High | Use baseline-only fallback, flag for manual review |
| Forecast accuracy worse than baseline | Medium | Medium | Test multiple Prophet configs, ensemble as hedge |
| GCP quota exceeded | Low | High | Monitor usage, set alerts, scale memory down if needed |
| Data quality issues (zeros, outliers) | High | Medium | Implement data validation, flag suspicious inputs |
| Client misinterprets recommendations | Medium | Medium | Provide clear documentation, confidence levels, FAQs |

## Rollout Plan

1. **Dev**: Test on TAOS sample (50 SKUs) — 1 week
2. **Staging**: Full TAOS + DAWBU datasets — 1 week
3. **Production**: Deploy weekly schedule, send client reports — ongoing
4. **Future**: Add new brands (FBSI, FBNN, etc.) — Q4 2026

## Resources

- **Developer**: 1 FTE (8 weeks)
- **Cloud Infrastructure**: GCP (BigQuery + Cloud Functions) — existing
- **Compute Cost**: $0/week (within free tier)
- **Storage**: ~100 MB BigQuery (forecasts + metrics)

## Deliverables Timeline

| Week | Deliverable | Status |
|------|-------------|--------|
| 1-2 | MVP core (segmentation + forecasting) | ⏳ |
| 3-4 | Validation + accuracy reporting | ⏳ |
| 5-6 | Recommendations + client reports | ⏳ |
| 7+ | Deployment + monitoring + scale | ⏳ |

---

**Owner**: Amit Mohanty  
**Status**: Planning  
**Last Updated**: Aug 14, 2026
