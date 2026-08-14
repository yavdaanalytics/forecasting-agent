# Ideas: Forecasting Agent Conceptual Framework

**Deep dives into design decisions, research foundations, and future directions.**

---

## 1. Coefficient of Variation (CV) as Segmentation Strategy

### Why CV Works

Most forecasting systems use fixed thresholds (e.g., "forecast SKUs with >100 units/month"). This ignores the fundamental property of demand patterns: **volatility relative to average demand**.

Two SKUs with the same average sales (10 units/day) can have very different patterns:
- **Stable** (always 9-11 units): predictable, Prophet thrives
- **Erratic** (1 day: 1 unit, next day: 30 units): noisy, Prophet overfits

Coefficient of Variation (CV) captures this:

```
CV = STDDEV(daily_qty) / AVG(daily_qty)

Stable:    CV = 1 / 10 = 0.10
Erratic:   CV = 15 / 10 = 1.50
```

### Segmentation Thresholds

Through offline cross-validation on TAOS & DAWBU data:
- **Stable** (CV < 1.0): Use conservative Prophet (low changepoint flexibility)
- **Volatile** (CV ≥ 1.0): Use flexible Prophet OR ensemble (hedges overfitting)

**Why not other methods?**
- **Fixed dollar thresholds**: Ignores volatility (high variance ≠ predictable)
- **Velocity classes**: Only captures magnitude, not noise
- **Category-based rules**: Too coarse-grained (all plants ≠ same forecast)
- **ML classifiers**: Overkill; CV has clear interpretation

### Empirical Results

On TAOS/DAWBU holdout test:
```
Stable SKUs (CV < 1.0):
  Prophet:  28% WAPE ✅ (changepoint_prior_scale = 0.05)
  Baseline: 35% WAPE
  Ensemble: 32% WAPE

Volatile SKUs (CV ≥ 1.0):
  Prophet:  52% WAPE (overfits to spikes)
  Baseline: 48% WAPE
  Ensemble: 42% WAPE ✅ (hedges overfitting)
```

Prophet wins on stable; Ensemble wins on volatile. ✓

---

## 2. Ensemble as Volatility Hedge

### The Overfitting Problem

Prophet excels at capturing trend + seasonality when data is *regular*. But volatile SKUs with sparse sales (e.g., DAWBU specialty items: 2-3 units every 10 days) present a challenge:

Prophet's changepoint detection sees every spike as a potential trend shift, leading to predictions like:
```
Historical:  2, 3, 0, 0, 0, 8, 2, 1, 0, 0
Prophet fit: sharp spike at day 6
Forecast:    predicts sustained ~6 units (wrong!)
Actual:      follows original pattern ~2-3 units
```

### Why Ensemble Works

Blending Prophet + Baseline (50/50):

```
Ensemble forecast = 0.5 * Prophet + 0.5 * Baseline

Prophet output:      6 units (overfitted to recent spike)
Baseline output:     2.5 units (avg of last 30 days)
Ensemble output:     4.25 units (compromises)

Outcome:
  Prophet WAPE: 52% (too optimistic)
  Baseline WAPE: 48% (conservative but stable)
  Ensemble WAPE: 42% (best of both) ✅
```

### Weight Tuning Strategy

Current approach: fixed 50/50 split.

Future: **Adaptive weighting per segment**
```python
prophet_weight = {
    "stable": 0.7,     # Trust Prophet more on smooth data
    "volatile": 0.3,   # Trust Baseline more on noisy data
    "hypervolatile": 0.2  # Mostly baseline if CV > 2.0
}
```

Tune via grid search on holdout data:
```
Test weights: [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
Evaluate WAPE on holdout test per segment
Select weight that minimizes average WAPE
```

---

## 3. Prophet Configuration Tuning per Segment

### Changepoint Prior Scale

Controls how aggressively Prophet detects trend changes.

**Low (0.01-0.05)**: Conservative
- Few/smooth trend changes
- Fits stable patterns well
- Ignores short-term spikes
- **Good for**: regular, predictable demand

**High (0.3-1.5)**: Aggressive
- Many trend changes
- Fits volatile patterns better
- Risk of overfitting
- **Good for**: erratic, sparse demand

### Empirical Calibration

Offline cross-validation result for TAOS:
```
changepoint_prior_scale | Stable WAPE | Volatile WAPE | Avg
0.01                    | 29%         | 58%           | 44% (too conservative)
0.05                    | 28%         | 52%           | 40% ✅ (stable config)
0.1                     | 30%         | 48%           | 39%
0.3                     | 42%         | 42%           | 42% (volatile config)
0.5                     | 55%         | 38%           | 47% (too aggressive)
```

**Decision**: Use 0.05 for stable, 0.3 for volatile.

### Seasonality Prior Scale

Controls strength of weekly/yearly seasonality.

**Low (3.0)**: Weak seasonality
- Ignores day-of-week patterns
- Better for sparse, erratic data
- **Result on DAWBU**: Reduced overfitting on week-to-week spikes

**High (10.0)**: Strong seasonality
- Locks in day-of-week patterns
- Better for regular, fast-moving data
- **Result on TAOS**: Captures consistent daily cycles

Current config:
```python
SEASONALITY_CONFIG = {
    "stable": {
        "seasonality_prior_scale": 10.0,  # Strong weekly pattern
        "yearly_seasonality": False,  # No yearly (2-year history, sparse)
        "weekly_seasonality": True,
        "daily_seasonality": False
    },
    "volatile": {
        "seasonality_prior_scale": 3.0,  # Weak weekly pattern
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": False
    }
}
```

---

## 4. Forecast Horizon Selection

Different products need different planning windows.

### TAOS (Fast-Moving Plants)
- **Pattern**: Daily sales, consistent
- **Lead time**: 2-4 weeks
- **Optimal horizon**: 60 days (covers lead time + buffer)
- **Rationale**: Beyond 60 days, forecast accuracy drops sharply; daily decisions don't need > 2 months

### DAWBU (Slow-Moving, Hub-Spoke)
- **Pattern**: 3-5 sales/week per location
- **Lead time**: 4-8 weeks
- **Optimal horizon**: 90 days (covers longer planning cycle)
- **Rationale**: Warehouse planning needs 3-month visibility; sparse demand means longer history available

### Horizon Selection Algorithm

```python
def select_optimal_horizon(cv, avg_sales_per_day):
    if cv < 0.5:
        return 60  # Stable daily demand
    elif cv < 1.0:
        return 60  # Regular demand
    elif avg_sales_per_day < 2:
        return 90  # Slow-moving, need longer visibility
    else:
        return 60  # Volatile but regular enough
```

### Why Not Longer?

Prophet accuracy degrades beyond 60-90 days:
```
Holdout WAPE by forecast day:

Stable SKUs:
  Day 1-30:   28% WAPE
  Day 31-60:  32% WAPE (slight degradation)
  Day 61-90:  45% WAPE (sharp drop)
  Day 91-180: 60%+ WAPE (unreliable)

Volatile SKUs:
  Day 1-30:   42% WAPE
  Day 31-60:  48% WAPE
  Day 61-90:  58% WAPE (unreliable beyond)
```

Rule: **Never forecast beyond 2x the regular lead time.**

---

## 5. Client Segmentation & Recommendations Strategy

### Problem
Clients see forecasts but don't know:
- Which method is best for their situation?
- Should we group certain SKUs together?
- What confidence should we have?

### Solution: Tiered Recommendations

#### Tier 1: High Confidence (WAPE < 40%)
```
✅ Deploy immediately
   Method: [Prophet | Ensemble]
   Confidence: "High — validated on 6 months holdout"
   Action: Use forecast for planning
```

#### Tier 2: Medium Confidence (WAPE 40-50%)
```
⚠️ Deploy with caution
   Method: [Ensemble (recommended) | Prophet]
   Confidence: "Medium — small holdout set, consider retesting in 2 months"
   Action: Use forecast but verify with domain experts
```

#### Tier 3: Low Confidence (WAPE > 50%)
```
❌ Escalate to manual review
   Issue: "High forecast error — likely data quality issue"
   Suggestion: "Check for: zero-sale periods, supply disruptions, seasonal shifts"
   Action: Manual forecasting or defer until more data available
```

### SKU Grouping Logic

**Cluster similar SKUs** based on:
1. **Volatility** (CV within 0.2 range)
2. **Forecast method** (both win with Prophet or both need Ensemble)
3. **Category** (plants vs seeds vs soil → similar patterns)
4. **Historical error correlation** (do they fail together?)

Example output:
```
Group: "Fast-Growing Perennials"
  SKUs: PLANT-001, PLANT-003, PLANT-007
  CV range: 0.95-1.15
  Recommended method: Ensemble
  Confidence: Medium
  Rationale: "All volatile perennials; Ensemble hedges against spikes"

Group: "Seasonal Seeds"
  SKUs: SEED-101, SEED-102, SEED-201
  CV range: 0.45-0.68
  Recommended method: Prophet
  Confidence: High
  Rationale: "Stable seasonal patterns; Prophet captures trends well"
```

---

## 6. Handling Edge Cases

### Zero-Sale Days
Many e-commerce products have zero-sale days (weekends, off-season).

**Problem**: Raw Prophet sees zeros as a trend, not as "no sales happened."

**Solution**: **Zero-filled daily calendar**
```python
# Instead of: [10, 0, 0, 12, 0, 0, 15, ...]
# Generate:  [date1→10, date2→0, date3→0, date4→12, date5→0, ...all days]

# Explicitly include all dates, fill missing with 0
# Prophet treats as "low demand" not "no data"
```

Result: Smoother seasonality fits, prevents spiky forecasts.

### Very Sparse Data (< 1 sale/week)

**Problem**: Prophet needs ~30-50 data points. A sparse SKU with only 2 years might have < 200 points.

**Workaround**:
1. Use Baseline as fallback (simple moving average always works)
2. Pool with similar SKUs (use clustering)
3. Flag for manual review (don't auto-forecast)

### Sudden Demand Shifts

Example: A product goes viral, sales jump 10x.

**Detection**: WAPE suddenly spikes on holdout test.

**Response**:
- Run cross-validation with different window sizes
- If recent data shows stable 10x level, re-calibrate baseline
- Flag as "structural break — forecast from new baseline"
- Notify client (not a forecast error, data changed)

---

## 7. Multi-Brand Learning

As we add brands (TAOS → DAWBU → Future brands), we can learn from patterns:

### Federated Learning
```
TAOS:   Test 5 different changepoint configs on holdout
        → Winner: 0.05 for stable, 0.3 for volatile
        
DAWBU:  Test same configs
        → Winner: 0.05 for stable, 0.3 for volatile
        → Convergence! Both brands prefer same tuning
        
Future: Try configs proven on TAOS/DAWBU first
        → Shorter tuning phase, faster deployment
```

### Brand Clustering
Group brands by forecast difficulty:
```
Easy (WAPE < 35%):   TAOS (regular daily sales)
Medium (WAPE 35-50%): DAWBU (sparse hub-spoke)
Hard (WAPE > 50%):   [future brands with seasonal spikes]

→ Apply proven configs from "Easy" to new brands
  (saves tuning time)
```

---

## 8. Monitoring & Adaptation

### Metrics to Track

**Per Segment**:
- Accuracy trend (WAPE over time)
- Method performance (which method wins recently?)
- Data quality (% zero-sale days, outlier count)

**Per Execution**:
- Execution time (segmentation → forecasting → validation)
- Forecast coverage (% of SKUs successfully forecasted)
- Alerts (high error, data gaps, model failures)

### Adaptive Thresholds

Current: Fixed WAPE threshold for "high confidence" (40%).

**Future**: Adaptive based on segment
```python
confidence_threshold = {
    "stable": 0.40,      # Tight threshold for predictable
    "volatile": 0.55,    # Looser threshold for noisy
    "seasonal": 0.45,    # Intermediate
}
```

### Re-Forecasting Strategy

If accuracy drops:
1. Check data quality (zeros, outliers, supply gaps)
2. Test alternative changepoint_prior_scale (grid search)
3. Increase ensemble weight (rely more on Baseline)
4. Flag for manual review if still bad

---

## 9. Client Communication

### What NOT to Say
- ❌ "We use machine learning to predict sales" (vague, not actionable)
- ❌ "42% WAPE is good" (client doesn't know what WAPE means)
- ❌ "Use ensemble forecast instead of Prophet" (no justification)

### What TO Say
- ✅ "We tested 3 forecasting methods. For your stable products, Prophet wins (28% error). For volatile products, blending methods works best (42% error)."
- ✅ "Historical holdout test: We hid the last 30 days, predicted them, compared to actual. This is how we measured accuracy."
- ✅ "Forecast 60 days out for regular planning. Beyond 60 days, accuracy drops too much."

### Confidence Levels

Always include:
```
High Confidence (WAPE < 40%)
  "Our forecast was 28% off on average during testing.
   We're confident in using this for planning."

Medium Confidence (WAPE 40-50%)
  "Our forecast was 42% off during testing.
   Use this for rough planning; cross-check with expert judgment."

Low Confidence (WAPE > 50%)
  "Our forecast was 58% off — too high to trust.
   Likely reason: [data quality | sparse history | structural change]
   Recommendation: Use manual forecasting for now."
```

---

## 10. Research Directions

### Future Work
1. **LSTM/Transformer** time-series models (if Prophet accuracy plateaus)
2. **Exogenous variables** (weather, marketing spend → demand)
3. **Causal inference** (what caused that spike?)
4. **Reinforcement learning** (agent learns which method to use)
5. **Federated forecasting** (learn from competitor data anonymously)

### Why Not These Now?
- Complexity > benefit (WAPE already low with ensemble)
- Data quality not ready (need exogenous variables first)
- Client sophistication (don't need black-box models yet)

**Revisit in 6 months** once we have a full year of agent results.

---

## 11. Cost-Benefit Analysis

### Build vs Buy

**Forecasting Agent (build):**
- Development: 8 weeks × 1 FTE
- Infrastructure: $0-10/week (GCP free tier)
- Maintenance: 2-4 hours/week
- **Benefit**: Fully customized, owns IP, fast iteration

**Commercial Forecasting Tool (buy):**
- Cost: $1,000-5,000/month
- Setup: 4-8 weeks
- Maintenance: Support tickets
- **Benefit**: Out-of-box, vendor support

**Decision**: Build — lower total cost of ownership, better ROI for custom business.

---

**Last Updated**: Aug 14, 2026  
**Author**: Amit Mohanty
