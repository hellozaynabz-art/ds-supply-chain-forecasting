"""Build a weekly demand panel: one row per (sku, warehouse, week).

* ``synthetic`` (default): 200 SKU-warehouse series over 3 years with trend, yearly
  seasonality, promotions and intermittent low-volume items.
* ``kaggle``: "Forecasts for Product Demand" (felixzhao/productdemandforecasting),
  aggregated to weeks; the 200 highest-volume series with a full history are kept.

Output: ``data/processed/weekly_demand.parquet`` with columns
    series_id, sku, warehouse, category, week, demand
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
N_SERIES = 200


def synthetic(n_series: int = N_SERIES, weeks: int = 156, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    t = np.arange(weeks)
    rows = []
    for i in range(n_series):
        category = f"Category_{rng.integers(1, 9):02d}"
        warehouse = rng.choice(["Whse_A", "Whse_C", "Whse_J", "Whse_S"])
        level = rng.lognormal(4.0, 1.0)
        trend = rng.normal(0.002, 0.004)
        amp = rng.uniform(0.0, 0.45)
        phase = rng.uniform(0, 2 * np.pi)
        mean = level * np.exp(trend * t) * (1 + amp * np.sin(2 * np.pi * t / 52 + phase))
        promo = rng.random(weeks) < 0.06
        mean = mean * np.where(promo, rng.uniform(1.4, 2.2), 1.0)
        demand = rng.negative_binomial(5, 5 / (5 + mean))
        if level < 25:  # slow movers: lots of zero weeks
            demand = demand * (rng.random(weeks) < 0.55)
        for w, d in zip(idx, demand, strict=True):
            rows.append((f"P{i:04d}", warehouse, category, w, int(d)))
    df = pd.DataFrame(rows, columns=["sku", "warehouse", "category", "week", "demand"])
    return _with_series_id(df)


def kaggle() -> pd.DataFrame:
    dest = RAW / "kaggle"
    path = dest / "Historical Product Demand.csv"
    if not path.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "uvx",
                "kaggle",
                "datasets",
                "download",
                "-d",
                "felixzhao/productdemandforecasting",
                "-p",
                str(dest),
                "--unzip",
            ],
            check=True,
        )
    df = pd.read_csv(path).dropna(subset=["Date"])
    # Returns are recorded as "(100)"; treat them as negative demand, then floor weekly totals at 0.
    demand = (
        df["Order_Demand"].astype(str).str.strip().str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    df = pd.DataFrame(
        {
            "sku": df["Product_Code"],
            "warehouse": df["Warehouse"],
            "category": df["Product_Category"],
            "week": pd.to_datetime(df["Date"]).dt.to_period("W-SUN").dt.start_time,
            "demand": pd.to_numeric(demand, errors="coerce").fillna(0),
        }
    )
    df = df[(df["week"] >= "2012-01-02") & (df["week"] < "2016-12-26")]
    weekly = df.groupby(["sku", "warehouse", "category", "week"], as_index=False)["demand"].sum()
    weekly["demand"] = weekly["demand"].clip(lower=0)
    top = (
        weekly.groupby(["sku", "warehouse"])
        .agg(weeks=("week", "nunique"), volume=("demand", "sum"))
        .query("weeks >= 200")
        .nlargest(N_SERIES, "volume")
        .index
    )
    weekly = weekly.set_index(["sku", "warehouse"]).loc[top].reset_index()
    # Fill weeks with no orders with zero demand.
    full = (
        weekly.set_index("week")
        .groupby(["sku", "warehouse", "category"])["demand"]
        .resample("W-MON")
        .sum()
        .reset_index()
    )
    return _with_series_id(full)


def _with_series_id(df: pd.DataFrame) -> pd.DataFrame:
    df["series_id"] = df["sku"] + "@" + df["warehouse"]
    return df[["series_id", "sku", "warehouse", "category", "week", "demand"]].sort_values(
        ["series_id", "week"], ignore_index=True
    )


def load() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "weekly_demand.parquet")


def build(source: str = "synthetic") -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = kaggle() if source == "kaggle" else synthetic()
    df.to_parquet(PROCESSED / "weekly_demand.parquet", index=False)
    print(f"[{source}] {df['series_id'].nunique()} series, {len(df):,} rows -> {PROCESSED}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    build(parser.parse_args().source)
