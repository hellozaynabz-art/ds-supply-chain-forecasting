"""Backtest the models and translate forecast accuracy into inventory outcomes.

Two forecast origins:
  * calibration origin (26 weeks before the end) -> per-series error spread for safety stock
  * evaluation origin (13 weeks before the end)  -> accuracy metrics + inventory simulation

    uv run python -m demand_forecast.backtest
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from demand_forecast import data, models  # noqa: E402

REPORTS = Path("reports")
HORIZON = 13
LEAD_TIME = 2  # weeks between placing and receiving an order
SERVICE_TARGET = 0.95


def forecast_all(panel: pd.DataFrame, origin: pd.Timestamp) -> dict[str, dict[str, np.ndarray]]:
    """Forecasts from every model for every series, using data strictly before ``origin``."""
    train = panel[panel["week"] < origin]
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, fn in models.LOCAL_MODELS.items():
        out[name] = {
            sid: fn(g["demand"].to_numpy(), HORIZON) for sid, g in train.groupby("series_id")
        }
    out["Global gradient boosting"] = models.global_gbm(train, HORIZON)
    return out


def actuals(panel: pd.DataFrame, origin: pd.Timestamp) -> dict[str, np.ndarray]:
    window = panel[
        (panel["week"] >= origin) & (panel["week"] < origin + pd.Timedelta(weeks=HORIZON))
    ]
    return {sid: g["demand"].to_numpy() for sid, g in window.groupby("series_id")}


def accuracy(
    fc: dict[str, np.ndarray], act: dict[str, np.ndarray], history: dict[str, np.ndarray]
) -> dict[str, float]:
    err = np.concatenate([fc[s] - act[s] for s in act])
    y = np.concatenate(list(act.values()))
    mase = []
    for s in act:
        h = history[s]
        scale = (
            np.mean(np.abs(h[models.SEASON :] - h[: -models.SEASON]))
            if len(h) > models.SEASON
            else 0
        )
        if scale > 0:
            mase.append(np.mean(np.abs(fc[s] - act[s])) / scale)
    return {
        "WAPE": np.abs(err).sum() / y.sum(),
        "Bias": err.sum() / y.sum(),
        "MASE": float(np.median(mase)),
    }


def simulate_inventory(
    demand: np.ndarray, forecast: np.ndarray, safety_stock: float
) -> tuple[float, float]:
    """Weekly order-up-to policy with lost sales. Returns (fill rate, average on-hand units)."""
    weeks = len(demand)
    fc = np.concatenate([forecast, np.repeat(forecast[-1], LEAD_TIME + 1)])
    target = lambda t: fc[t : t + LEAD_TIME + 1].sum() + safety_stock  # noqa: E731
    on_hand = target(0)
    pipeline = [0.0] * LEAD_TIME
    filled = held = 0.0
    for t in range(weeks):
        on_hand += pipeline.pop(0)
        sold = min(on_hand, demand[t])
        filled += sold
        on_hand -= sold
        held += on_hand
        position = on_hand + sum(pipeline)
        pipeline.append(max(0.0, target(t + 1) - position))
    total = demand.sum()
    return (filled / total if total else 1.0), held / weeks


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    panel = data.load()
    end = panel["week"].max() + pd.Timedelta(weeks=1)
    calib_origin = end - pd.Timedelta(weeks=2 * HORIZON)
    eval_origin = end - pd.Timedelta(weeks=HORIZON)

    print("forecasting from calibration origin…")
    calib_fc, calib_act = forecast_all(panel, calib_origin), actuals(panel, calib_origin)
    print("forecasting from evaluation origin…")
    eval_fc, eval_act = forecast_all(panel, eval_origin), actuals(panel, eval_origin)
    history = {
        sid: g["demand"].to_numpy()
        for sid, g in panel[panel["week"] < eval_origin].groupby("series_id")
    }

    rows = []
    z = norm.ppf(SERVICE_TARGET)
    for name in eval_fc:
        acc = accuracy(eval_fc[name], eval_act, history)
        fill, stock = [], []
        for sid, act in eval_act.items():
            sigma = np.std(calib_fc[name][sid] - calib_act[sid])
            ss = z * sigma * np.sqrt(LEAD_TIME + 1)
            f, s = simulate_inventory(act, eval_fc[name][sid], ss)
            fill.append(f * act.sum())
            stock.append(s)
        total = sum(a.sum() for a in eval_act.values())
        rows.append(
            {
                "model": name,
                **acc,
                "fill_rate": sum(fill) / total,
                "avg_inventory_units": sum(stock),
            }
        )
    results = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    best = results.iloc[0]
    naive = results.set_index("model").loc["Moving average (8w)"]  # the usual spreadsheet method

    _plot_examples(panel, eval_origin, eval_fc, best["model"])
    _plot_tradeoff(results)

    fmt = results.copy()
    for c in ["WAPE", "Bias", "fill_rate"]:
        fmt[c] = fmt[c].map("{:.1%}".format)
    fmt["MASE"] = fmt["MASE"].map("{:.2f}".format)
    fmt["avg_inventory_units"] = fmt["avg_inventory_units"].map("{:,.0f}".format)
    inv_change = best["avg_inventory_units"] / naive["avg_inventory_units"] - 1

    report = f"""# Demand Forecasting & Inventory Backtest

_Generated by `make backtest`._ {panel["series_id"].nunique()} SKU-warehouse series,
weekly data from {panel["week"].min():%b %Y} to {panel["week"].max():%b %Y}.
Forecasts are made {HORIZON} weeks ahead from {eval_origin:%d %b %Y} using only earlier data.

## Accuracy

{fmt.to_markdown(index=False)}

- **WAPE**: total absolute error as a share of total demand (lower is better)
- **Bias**: positive means over-forecasting
- **MASE**: error relative to a same-week-last-year forecast, median across series (below 1 beats it)

## From accuracy to inventory

Each model drives a weekly order-up-to policy with a {LEAD_TIME}-week lead time and safety stock sized
for a {SERVICE_TARGET:.0%} service target from that model's own historical error.

![tradeoff](tradeoff.png)

**{best["model"]}** reaches a {best["fill_rate"]:.1%} fill rate while holding
**{abs(inv_change):.0%} {"less" if inv_change < 0 else "more"} inventory** than an 8-week moving average,
the method most planning spreadsheets use ({naive["fill_rate"]:.1%} fill rate).

![examples](examples.png)
"""
    (REPORTS / "REPORT.md").write_text(report)
    print(report)


def _plot_examples(panel, origin, fc, model_name) -> None:
    ids = (
        panel.groupby("series_id")["demand"]
        .sum()
        .sort_values(ascending=False)
        .index[[0, 20, 60, 120]]
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, sid in zip(axes.flat, ids, strict=True):
        g = panel[(panel["series_id"] == sid) & (panel["week"] >= origin - pd.Timedelta(weeks=52))]
        ax.plot(g["week"], g["demand"], color="#495057", lw=1, label="Actual")
        weeks = pd.date_range(origin, periods=HORIZON, freq="W-MON")
        ax.plot(weeks, fc[model_name][sid], color="#4C6EF5", lw=2, label=model_name)
        ax.plot(
            weeks, fc["Seasonal naive"][sid], color="#F08C00", lw=1, ls="--", label="Seasonal naive"
        )
        ax.axvline(origin, color="grey", lw=0.8)
        ax.set_title(sid, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Example forecasts (last 52 weeks + 13-week horizon)")
    fig.tight_layout()
    fig.savefig(REPORTS / "examples.png", dpi=150)
    plt.close(fig)


def _plot_tradeoff(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(results["avg_inventory_units"], results["fill_rate"], s=60, color="#4C6EF5")
    for _, r in results.iterrows():
        ax.annotate(
            r["model"],
            (r["avg_inventory_units"], r["fill_rate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Average inventory held (units)")
    ax.set_ylabel("Fill rate")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.set_title("Service vs. inventory by forecasting model (up and left is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
