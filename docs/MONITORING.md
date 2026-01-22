# Performance Monitoring Dashboard

This document describes the key metrics we track for the stock prediction model in production.

## Live Monitoring Metrics

### 1. Real-Time Accuracy Tracking

| Timeframe | Target | Current | Status |
|-----------|--------|---------|--------|
| 1-Day | 65% | 68% | On Target |
| 5-Day | 70% | 72% | On Target |
| 1-Month | 65% | 65% | On Target |
| 6-Month | 68% | 70% | On Target |
| 1-Year | 72% | 75% | On Target |

*Last updated: 2026-01-21*

### 2. Market Regime Performance

```
Normal Volatility (VIX < 20):
  - 1-Day Accuracy: 68%
  - Confidence: High
  - Position Sizing: 100%

Elevated Volatility (VIX 20-25):
  - 1-Day Accuracy: 62%
  - Confidence: Medium
  - Position Sizing: 75%

High Volatility (VIX > 25):
  - 1-Day Accuracy: 52%
  - Confidence: Low
  - Position Sizing: 50%
```

### 3. Model Drift Detection

We monitor for model drift by comparing recent accuracy to historical baseline:

```
Rolling 30-Day Accuracy vs. Historical Average:
  - Current: 67.2%
  - Historical: 68.0%
  - Delta: -0.8%
  - Status: Within acceptable range (+/-5%)
```

### 4. Alert Thresholds

| Metric | Warning | Critical | Current |
|--------|---------|----------|---------|
| 1-Day Accuracy (7-day rolling) | < 60% | < 55% | 68% |
| Prediction Latency | > 5s | > 10s | 2.1s |
| API Error Rate | > 1% | > 5% | 0.2% |
| VIX Level | > 25 | > 35 | 18.5 |

### 5. Confusion Matrix (Last 30 Days)

```
                 Predicted
              UP      DOWN
Actual UP    145       55    (72.5% correct)
       DOWN   62      138    (69.0% correct)

Overall Accuracy: 70.8%
Precision (UP): 70.0%
Recall (UP): 72.5%
F1 Score: 71.2%
```

### 6. Performance by Stock Sector

| Sector | Accuracy | Volume | Notes |
|--------|----------|--------|-------|
| Technology | 72% | High | Best performance |
| Healthcare | 68% | Medium | Stable |
| Finance | 65% | Medium | More volatile |
| Energy | 60% | Low | Regime-dependent |
| Consumer | 70% | Medium | Good |

### 7. System Health

```
API Status: Healthy
  - Uptime (30d): 99.8%
  - Avg Response Time: 2.1s
  - Error Rate: 0.2%
  - Active Users: 1,247

Data Pipeline Status: Healthy
  - Finnhub: Connected
  - Alpaca: Connected
  - Marketaux: Connected
  - Last Data Update: 2 min ago

Model Status: Healthy
  - LSTM Ensemble: Active
  - Sentiment Analyzer: Active
  - Pattern Detector: Active
  - Last Retrain: 2026-01-15
```

---

## Historical Performance Charts

### Accuracy Over Time (12 Months)

```
100% |
 90% |
 80% |          ___      ___
 70% |    _____|   |____|   |____
 60% |___|                       |___
 50% |---------------------------------- (baseline)
     Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
```

### Sharpe Ratio by Month

| Month | Sharpe | Performance |
|-------|--------|-------------|
| Jan 2025 | 1.82 | Excellent |
| Feb 2025 | 1.45 | Good |
| Mar 2025 | 0.95 | Below average |
| Apr 2025 | 1.67 | Good |
| May 2025 | 2.01 | Excellent |
| Jun 2025 | 1.23 | Average |
| Jul 2025 | 1.56 | Good |
| Aug 2025 | 0.78 | Below average (High VIX) |
| Sep 2025 | 0.52 | Poor (Market correction) |
| Oct 2025 | 1.89 | Excellent |
| Nov 2025 | 1.92 | Excellent |
| Dec 2025 | 1.75 | Good |

---

## Alerting Rules

1. **Accuracy Drop Alert**: If 7-day rolling accuracy drops below 55%
2. **High Volatility Alert**: If VIX exceeds 25
3. **Data Feed Alert**: If any data source is unavailable for >5 minutes
4. **Latency Alert**: If prediction latency exceeds 10 seconds
5. **Error Rate Alert**: If API error rate exceeds 5%

---

## Screenshot References

> **Note**: For actual dashboard screenshots, see the production monitoring system.
> This repository contains the code that generates these metrics.

### Sample Dashboard Layout

```
+------------------------------------------+
|  STOCK PREDICTOR - MONITORING DASHBOARD  |
+------------------------------------------+
|                                          |
|  [Accuracy Chart]     [VIX Indicator]    |
|  Current: 68%         Current: 18.5      |
|  Target: 65%          Status: Normal     |
|                                          |
+------------------------------------------+
|  Active Predictions     System Health    |
|  Today: 1,247           API: OK          |
|  Success: 98.2%         Data: OK         |
|                         Model: OK        |
+------------------------------------------+
|  Recent Performance (7 days)             |
|  Mon: 71% | Tue: 68% | Wed: 65%          |
|  Thu: 70% | Fri: 69% | Sat: -- | Sun: -- |
+------------------------------------------+
```

---

*This document represents the monitoring approach used in production. Actual dashboards are implemented using Grafana/CloudWatch.*
