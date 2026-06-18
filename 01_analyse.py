"""
Demand Forecasting and Inventory Optimisation
=============================================
Real data: Store Item Demand Forecasting Challenge (Kaggle), 10 stores x 50 items,
daily sales 2013-01-01 to 2017-12-31 (913,000 rows).

Pipeline:
  1. Exploratory analysis  (trend, weekly + yearly seasonality, store/item structure)
  2. Forecasting backtest  (90-day holdout): Naive, Seasonal-naive, ETS, Global ML
  3. Inventory optimisation (safety stock, reorder point, EOQ) + a 90-day
     reorder-point simulation comparing a forecast-driven policy to a
     historical-variability policy, at a fixed 95% service-level target.

Outputs: figures + outputs/metrics.json
Run: python 01_analyse.py
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# palette (warm, no default blue)
INK = "#1d3a34"; TEAL = "#2a9d8f"; GOLD = "#e9b949"; CORAL = "#e76f51"
MUT = "#9ec7bd"; GREY = "#b9b9b9"
plt.rcParams.update({"axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 110})

HORIZON = 90          # forecast / holdout length (days), matches the challenge
LEAD_TIME = 7         # replenishment lead time (days)
SERVICE = 0.95        # target cycle service level
Z = 1.6449            # z-score for 95% service level
UNIT_COST = 8.0       # assumed cost of goods per unit ($)
HOLDING_RATE = 0.25   # assumed annual holding cost as a fraction of unit cost
ORDER_COST = 60.0     # assumed fixed cost per replenishment order ($)

print("Loading data ...")
df = pd.read_csv("data/train.csv", parse_dates=["date"])
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
cutoff = df["date"].max() - pd.Timedelta(days=HORIZON - 1)   # first day of holdout
print(f"  rows={len(df):,}  dates {df.date.min().date()}..{df.date.max().date()}")
print(f"  holdout = last {HORIZON} days (from {cutoff.date()})")

metrics = {"data": {"rows": int(len(df)), "stores": int(df.store.nunique()),
                    "items": int(df.item.nunique()),
                    "date_min": str(df.date.min().date()),
                    "date_max": str(df.date.max().date()),
                    "series": int(df.store.nunique() * df.item.nunique())},
           "config": {"horizon_days": HORIZON, "lead_time_days": LEAD_TIME,
                      "service_level": SERVICE, "unit_cost": UNIT_COST,
                      "holding_rate": HOLDING_RATE, "order_cost": ORDER_COST}}

# ----------------------------------------------------------------------------
# 1. EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------------
print("Exploratory analysis ...")
daily = df.groupby("date", as_index=False)["sales"].sum()

# Fig 1: total daily sales + 30-day moving average (trend)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(daily.date, daily.sales, color=MUT, lw=0.7, label="Daily total")
ax.plot(daily.date, daily.sales.rolling(30, center=True).mean(),
        color=INK, lw=2.2, label="30-day average")
ax.set_title("Total daily units sold across all stores and items (2013-2017)")
ax.set_xlabel(""); ax.set_ylabel("Units sold"); ax.legend(loc="upper left")
yr = daily.groupby(daily.date.dt.year)["sales"].sum()
growth = (yr.iloc[-1] / yr.iloc[0]) ** (1 / (len(yr) - 1)) - 1
ax.annotate(f"Demand grows about {growth*100:.0f}% per year",
            xy=(0.02, 0.92), xycoords="axes fraction", color=CORAL, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig1_trend.png"); plt.close(fig)
metrics["eda"] = {"avg_annual_growth_pct": round(growth * 100, 1),
                  "total_units_2017": int(yr.iloc[-1])}

# Fig 2: weekly and monthly seasonality
d = df.copy()
d["dow"] = d.date.dt.dayofweek; d["month"] = d.date.dt.month
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
wk = d.groupby("dow")["sales"].mean()
mo = d.groupby("month")["sales"].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(range(7), wk.values, color=[TEAL if i < 5 else GOLD for i in range(7)])
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(dow_names)
axes[0].set_title("Weekly pattern: weekends sell more")
axes[0].set_ylabel("Average units / store-item / day")
axes[1].plot(range(1, 13), mo.values, "-o", color=CORAL, lw=2)
axes[1].set_xticks(range(1, 13))
axes[1].set_title("Seasonal pattern: peaks in summer")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Average units")
fig.tight_layout(); fig.savefig(OUT / "fig2_seasonality.png"); plt.close(fig)
metrics["eda"]["weekend_uplift_pct"] = round(
    (wk[[5, 6]].mean() / wk[[0, 1, 2, 3, 4]].mean() - 1) * 100, 1)
metrics["eda"]["peak_month"] = int(mo.idxmax())

# Fig 3: demand structure across stores and items (heatmap of mean daily sales)
piv = df.groupby(["store", "item"])["sales"].mean().unstack("item")
fig, ax = plt.subplots(figsize=(11, 4.2))
sns.heatmap(piv, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Mean units/day"})
ax.set_title("Average daily demand varies widely by store and item")
ax.set_xlabel("Item"); ax.set_ylabel("Store")
fig.tight_layout(); fig.savefig(OUT / "fig3_store_item.png"); plt.close(fig)
store_tot = df.groupby("store")["sales"].sum()
metrics["eda"]["store_demand_ratio_max_min"] = round(
    store_tot.max() / store_tot.min(), 2)

# ----------------------------------------------------------------------------
# 2. FORECASTING BACKTEST (90-day holdout)
# ----------------------------------------------------------------------------
print("Building features for the global ML model ...")
# Wide panel keyed by (store,item); features chosen so they never peek inside the
# 90-day horizon (all lags >= HORIZON), giving an honest h-step-ahead forecast.
g = df.groupby(["store", "item"], group_keys=False)
feat = df.copy()
feat["lag_90"]  = g["sales"].shift(90)
feat["lag_180"] = g["sales"].shift(180)
feat["lag_364"] = g["sales"].shift(364)     # same day, previous year
feat["lag_728"] = g["sales"].shift(728)     # same day, two years prior
# trailing 28-day mean ending 90 days back, and the same window a year earlier
feat["roll_90"]  = g["sales"].shift(90).rolling(28).mean().reset_index(0, drop=True)
feat["roll_364"] = g["sales"].shift(364).rolling(28).mean().reset_index(0, drop=True)
feat["dow"] = feat.date.dt.dayofweek
feat["month"] = feat.date.dt.month
feat["day"] = feat.date.dt.day
feat["doy"] = feat.date.dt.dayofyear
feat["year"] = feat.date.dt.year
feat["weekend"] = (feat.dow >= 5).astype(int)

FEATS = ["store", "item", "lag_90", "lag_180", "lag_364", "lag_728",
         "roll_90", "roll_364", "dow", "month", "day", "doy", "year", "weekend"]
train = feat[(feat.date < cutoff) & feat[FEATS].notna().all(axis=1)]
test = feat[feat.date >= cutoff].copy()

print(f"  training global model on {len(train):,} rows ...")
model = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, max_depth=8,
    l2_regularization=1.0, random_state=42)
model.fit(train[FEATS], train["sales"])
test["pred_ml"] = np.clip(model.predict(test[FEATS]), 0, None)

# Baselines on the same holdout -----------------------------------------------
hist = df[df.date < cutoff]
last_val = hist.groupby(["store", "item"])["sales"].last().rename("naive")
# seasonal naive (yearly): actual value 364 days before each holdout date
sn = feat[["store", "item", "date", "lag_364"]].copy()
test = test.merge(last_val, on=["store", "item"], how="left")
test = test.merge(sn.rename(columns={"lag_364": "pred_snaive"}),
                  on=["store", "item", "date"], how="left")
test["pred_naive"] = test["naive"]

# ETS (Holt-Winters: additive trend + weekly seasonality) on a representative
# stratified sample of series (fast; classical models are fit per series).
sample = [(s, i) for s in range(1, 11) for i in range(1, 51, 4)]   # 130 series
print(f"  fitting ETS on {len(sample)} representative series ...")
ets_rows = []
for s, i in sample:
    y = hist[(hist.store == s) & (hist.item == i)].set_index("date")["sales"].asfreq("D")
    try:
        fit = ExponentialSmoothing(y, trend="add", seasonal="add",
                                   seasonal_periods=7,
                                   initialization_method="estimated").fit()
        fc = fit.forecast(HORIZON).clip(lower=0)
        ets_rows.append(pd.DataFrame({"store": s, "item": i,
                                      "date": fc.index, "pred_ets": fc.values}))
    except Exception:
        pass
ets_df = pd.concat(ets_rows, ignore_index=True)
test = test.merge(ets_df, on=["store", "item", "date"], how="left")


def wmape(a, f):
    return np.abs(a - f).sum() / np.abs(a).sum() * 100


# Evaluate every model on the SAME ETS sample for a fair comparison
mask = test.set_index(["store", "item"]).index.isin(sample)
ev = test[mask]
results = {}
for name, col in [("Naive", "pred_naive"), ("Seasonal naive", "pred_snaive"),
                  ("ETS", "pred_ets"), ("Global ML", "pred_ml")]:
    sub = ev.dropna(subset=[col])
    results[name] = {"MAE": round(mean_absolute_error(sub.sales, sub[col]), 2),
                     "WMAPE_pct": round(wmape(sub.sales, sub[col]), 2)}
    print(f"    {name:16s} MAE={results[name]['MAE']:6.2f}  "
          f"WMAPE={results[name]['WMAPE_pct']:5.2f}%")
metrics["forecast"] = {"models": results,
                       "eval_series": len(sample), "horizon_days": HORIZON}
best = min(results, key=lambda k: results[k]["WMAPE_pct"])
metrics["forecast"]["best_model"] = best
improve = (1 - results["Global ML"]["WMAPE_pct"] /
           results["Seasonal naive"]["WMAPE_pct"]) * 100
metrics["forecast"]["ml_vs_snaive_error_reduction_pct"] = round(improve, 1)

# Fig 4: model comparison
fig, ax = plt.subplots(figsize=(8, 4.4))
names = list(results)
vals = [results[n]["WMAPE_pct"] for n in names]
cols = [GREY, MUT, GOLD, TEAL]
bars = ax.bar(names, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
            ha="center", fontweight="bold")
ax.set_title(f"Forecast accuracy on a {HORIZON}-day holdout (lower is better)")
ax.set_ylabel("WMAPE  (weighted mean abs. % error)")
fig.tight_layout(); fig.savefig(OUT / "fig4_model_comparison.png"); plt.close(fig)

# Fig 5: example forecast vs actual for one representative series
s0, i0 = 2, 25
recent = df[(df.store == s0) & (df.item == i0) & (df.date >= cutoff - pd.Timedelta(days=120))]
fc_ml = test[(test.store == s0) & (test.item == i0)]
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(recent.date, recent.sales, color=INK, lw=1.4, label="Actual")
ax.plot(fc_ml.date, fc_ml.pred_ml, color=CORAL, lw=2, label="Global ML forecast")
ax.axvline(cutoff, color=GREY, ls="--")
ax.axvspan(cutoff, df.date.max(), color=GOLD, alpha=0.10)
ax.set_title(f"Forecast vs actual, store {s0} / item {i0} (90-day holdout shaded)")
ax.set_xlabel(""); ax.set_ylabel("Units sold"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig5_forecast_example.png"); plt.close(fig)

# ----------------------------------------------------------------------------
# 3. INVENTORY OPTIMISATION
# ----------------------------------------------------------------------------
print("Inventory optimisation ...")
# Per-series forecast-error std (from the ML holdout) vs raw historical demand std.
err = (test.assign(e=test.sales - test.pred_ml)
       .groupby(["store", "item"])["e"].std().rename("sigma_fcast"))
hist_mu = hist.groupby(["store", "item"])["sales"].mean().rename("mu")
hist_sd = hist.groupby(["store", "item"])["sales"].std().rename("sigma_hist")
inv = pd.concat([hist_mu, hist_sd, err], axis=1).dropna()

# Safety stock over the lead time, two ways:
# We use a periodic-review, order-up-to (R, S) policy: every R days the stock is
# topped up to a level S. The protection window is the review period plus the
# lead time, so safety stock is the deciding lever (this is how most retailers
# actually reorder, and it isolates the value of forecast quality).
REVIEW = 7                       # order once a week
prot = REVIEW + LEAD_TIME        # protection period (days)
inv["ss_fcast"] = Z * inv.sigma_fcast * np.sqrt(prot)
inv["ss_hist"] = Z * inv.sigma_hist * np.sqrt(prot)
inv["S_fcast"] = inv.mu * prot + inv.ss_fcast      # order-up-to level
inv["S_hist"] = inv.mu * prot + inv.ss_hist
# Economic order quantity (reported as a classic sizing benchmark)
annual_D = inv.mu * 365
H = UNIT_COST * HOLDING_RATE
inv["eoq"] = np.sqrt(2 * annual_D * ORDER_COST / H)

metrics["inventory"] = {
    "lead_time_days": LEAD_TIME, "service_target": SERVICE,
    "avg_safety_stock_forecast": round(inv.ss_fcast.mean(), 1),
    "avg_safety_stock_historical": round(inv.ss_hist.mean(), 1),
    "safety_stock_reduction_pct": round(
        (1 - inv.ss_fcast.mean() / inv.ss_hist.mean()) * 100, 1),
    "avg_eoq_units": round(inv.eoq.mean(), 1)}

# Fig 6: safety stock, forecast-driven vs historical-variability
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar(["Historical\nvariability", "Forecast-driven\n(ML)"],
       [inv.ss_hist.mean(), inv.ss_fcast.mean()], color=[GREY, TEAL])
for x, v in zip([0, 1], [inv.ss_hist.mean(), inv.ss_fcast.mean()]):
    ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_title(f"Average safety stock per series at {int(SERVICE*100)}% service\n"
             "(better forecasts need a smaller buffer)")
ax.set_ylabel("Safety-stock units")
fig.tight_layout(); fig.savefig(OUT / "fig6_safety_stock.png"); plt.close(fig)

# ----- 90-day reorder-point simulation, both policies, same service target ----
print("  simulating reorder policies over the holdout ...")
actual = (test.pivot_table(index="date", columns=["store", "item"],
                           values="sales").sort_index())
dates = actual.index


def simulate(S_map):
    """Periodic-review order-up-to (R,S): every REVIEW days top up to level S."""
    tot_demand = tot_short = 0.0
    inv_levels = []
    for (s, i) in actual.columns:
        S = S_map[(s, i)]
        on_hand = S                        # start full
        pipeline = {}                      # arrival_day -> qty
        for t, dt in enumerate(dates):
            on_hand += pipeline.pop(t, 0)
            dem = actual.loc[dt, (s, i)]
            shipped = min(on_hand, dem)
            tot_demand += dem; tot_short += dem - shipped
            on_hand -= shipped
            inv_levels.append(on_hand)
            if t % REVIEW == 0:            # review day: order up to S
                position = on_hand + sum(pipeline.values())
                order = max(S - position, 0)
                if order > 0:
                    pipeline[t + LEAD_TIME] = pipeline.get(t + LEAD_TIME, 0) + order
    fill = 1 - tot_short / tot_demand
    return fill, float(np.mean(inv_levels))


fill_f, invlvl_f = simulate(inv.S_fcast.to_dict())
fill_h, invlvl_h = simulate(inv.S_hist.to_dict())

# Annual holding cost across the whole network (all 500 series)
def hold_cost(level):
    return level * len(actual.columns) * UNIT_COST * HOLDING_RATE


cost_f, cost_h = hold_cost(invlvl_f), hold_cost(invlvl_h)
savings = cost_h - cost_f
metrics["inventory"].update({
    "sim_fill_rate_forecast": round(fill_f * 100, 2),
    "sim_fill_rate_historical": round(fill_h * 100, 2),
    "sim_avg_inventory_forecast": round(invlvl_f, 2),
    "sim_avg_inventory_historical": round(invlvl_h, 2),
    "annual_holding_cost_forecast": round(cost_f, 0),
    "annual_holding_cost_historical": round(cost_h, 0),
    "annual_holding_savings": round(savings, 0),
    "holding_savings_pct": round(savings / cost_h * 100, 1)})
print(f"    forecast policy : fill={fill_f*100:.1f}%  avg inv={invlvl_f:.1f}")
print(f"    historical policy: fill={fill_h*100:.1f}%  avg inv={invlvl_h:.1f}")
print(f"    annual holding savings: ${savings:,.0f}")

# Fig 7: simulation outcome (service achieved vs inventory held)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].bar(["Historical", "Forecast-driven"], [fill_h * 100, fill_f * 100],
            color=[GREY, TEAL])
axes[0].axhline(SERVICE * 100, color=CORAL, ls="--", label="95% target")
axes[0].set_ylim(85, 101); axes[0].set_title("Service level achieved")
axes[0].set_ylabel("Fill rate (%)"); axes[0].legend(loc="lower right")
for x, v in enumerate([fill_h * 100, fill_f * 100]):
    axes[0].text(x, v + 0.2, f"{v:.1f}%", ha="center", fontweight="bold")
axes[1].bar(["Historical", "Forecast-driven"], [invlvl_h, invlvl_f],
            color=[GREY, TEAL])
axes[1].set_title("Average inventory held per series")
axes[1].set_ylabel("Units on hand")
for x, v in enumerate([invlvl_h, invlvl_f]):
    axes[1].text(x, v + 0.3, f"{v:.1f}", ha="center", fontweight="bold")
fig.suptitle("Same service target, less stock: the value of a better forecast",
             fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig7_simulation.png"); plt.close(fig)

# Fig 8: where the demand and the inventory dollars concentrate (Pareto)
ann = (inv.mu * 365 * UNIT_COST).sort_values(ascending=False).reset_index(drop=True)
cum = ann.cumsum() / ann.sum() * 100
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(np.arange(1, len(cum) + 1), cum, color=INK, lw=2)
p20 = cum.iloc[int(len(cum) * 0.2)]
ax.axvline(len(cum) * 0.2, color=CORAL, ls="--")
ax.axhline(p20, color=CORAL, ls="--")
ax.annotate(f"Top 20% of store-items\n= {p20:.0f}% of revenue",
            xy=(len(cum) * 0.2, p20), xytext=(len(cum) * 0.3, p20 - 18),
            color=CORAL, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CORAL))
ax.set_title("Demand is concentrated: focus inventory effort on the vital few")
ax.set_xlabel("Store-item combinations (ranked)")
ax.set_ylabel("Cumulative revenue (%)")
fig.tight_layout(); fig.savefig(OUT / "fig8_pareto.png"); plt.close(fig)
metrics["inventory"]["top20pct_revenue_share"] = round(p20, 1)

json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
print("\nSaved 8 figures + outputs/metrics.json")
print(json.dumps(metrics["forecast"], indent=2))
print(json.dumps(metrics["inventory"], indent=2))
