# Demand Forecasting and Inventory Optimisation

Forecasting daily sales for **500 store-item combinations**, then turning the
forecast into a stocking policy that **cuts inventory about 13% while still
meeting a 95% service target**. Built in **Python** from a real, open dataset.

> A global gradient-boosting model forecasts demand with **11.4% error (WMAPE)**
> on a 90-day holdout, about **28% more accurate** than a strong seasonal
> benchmark. Sizing safety stock from that forecast (instead of from raw demand
> swings) roughly **halves the buffer** and, in a day-by-day reorder simulation,
> holds **13% less inventory** at the same service level.

---

## Read it

**[Full visual summary (web)](https://kingsley-amg.github.io/demand-forecasting/)**, with a
plain-English explanation under every chart. A **PDF report** is in [`report/`](report/).

## The problem

Retailers lose money two ways at once: stockouts (lost sales) and overstock
(frozen cash, markdowns, waste). The lever that controls both is demand
forecast accuracy, and how well that forecast is turned into reorder decisions.

## What it does

1. **Exploratory analysis** of five years of daily sales: an upward trend
   (about 7.8% per year), a weekly rhythm (weekends about 23% busier) and a
   summer seasonal peak, plus wide variation across stores and items.
2. **Forecasting backtest** on the last 90 days, comparing four models:
   - Naive (repeat last value)
   - Seasonal naive (repeat the value from a year ago)
   - ETS (classical trend plus weekly seasonality, statsmodels)
   - **Global gradient boosting** (`HistGradientBoostingRegressor`) trained
     across all 500 series with calendar features and lags of 90+ days, so it
     never peeks into the forecast window.
3. **Inventory optimisation**: convert each forecast into a weekly
   order-up-to policy, size safety stock for a 95% service target, report the
   economic order quantity, and run a day-by-day reorder **simulation** that
   compares a forecast-driven policy with a historical-variability policy.

## Key results

| Model | WMAPE (90-day holdout) |
|---|---|
| Naive | 25.2% |
| Seasonal naive | 15.8% |
| ETS | 19.5% |
| **Global ML** | **11.4%** |

- Safety stock needed for 95% service: **cut about 50%** by using forecast error
  instead of raw demand variability.
- Simulation across all 500 store-items: forecast-driven policy meets the 95%
  target with **13% less average inventory** than the historical policy (which
  reaches 99.6% only by overstocking).
- Demand is concentrated: the top 20% of store-items drive about 34% of revenue.

## Recommendations

1. Adopt the ML forecast for replenishment (about 28% more accurate, one model
   serves all 500 series).
2. Re-size safety stock from forecast error, freeing working capital at the same
   service promise.
3. Differentiate service by item importance; watch the vital few most closely.
4. Pilot, measure realised service and inventory for a quarter, then scale, and
   re-train as demand shifts.

## Structure

```
demand-forecasting/
|- 01_analyse.py        # EDA, forecasting backtest, inventory optimisation + simulation
|- 02_make_report.py    # builds the PDF report
|- data/                # train.csv (Kaggle Store Item Demand Challenge)
|- outputs/             # 8 figures + metrics.json
|- docs/index.html      # visual summary (GitHub Pages)
|- report/              # PDF report
```

## Reproduce

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn reportlab
python 01_analyse.py
python 02_make_report.py
```

## Data and tools

Python, pandas, scikit-learn (HistGradientBoosting), statsmodels (ETS),
matplotlib and seaborn. Data: the **Store Item Demand Forecasting Challenge**
dataset (Kaggle), 10 stores, 50 items, daily sales 2013-2017 (913,000 rows).

## Honest limitations

This is a clean benchmark dataset, so accuracy is higher than messy real data
would allow. The cost figures rest on assumed unit, holding and order costs; the
percentage improvements do not depend on those assumptions. Lead time is treated
as fixed; real lead-time variability would call for a slightly larger buffer. A
live pilot should confirm the gains before full rollout.

## Author

**Kingsley Amegah**, Data Scientist. GitHub: [@Kingsley-amg](https://github.com/Kingsley-amg)
