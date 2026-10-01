"""Forecasting models. Each takes the training history and returns ``horizon`` weekly forecasts.

Local models fit one series at a time; the global model learns across all series at once.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON = 52
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]


def seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Same week last year; falls back to the last value for short histories."""
    if len(history) >= SEASON:
        return history[-SEASON:][:horizon].astype(float)
    return np.repeat(float(history[-1]), horizon)


def moving_average(history: np.ndarray, horizon: int, window: int = 8) -> np.ndarray:
    return np.repeat(history[-window:].mean(), horizon)


def holt_winters(history: np.ndarray, horizon: int) -> np.ndarray:
    """Additive trend (damped) and seasonality; drops seasonality for short or sparse series."""
    y = history.astype(float)
    seasonal = "add" if len(y) >= 2 * SEASON and (y > 0).mean() > 0.8 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=SEASON if seasonal else None,
        ).fit()
    return np.clip(model.forecast(horizon), 0, None)


def _features(panel: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Lag features on demand divided by each series' average, so one model fits all sizes."""
    df = panel.sort_values(["series_id", "week"]).copy()
    df["y"] = df["demand"] / df["series_id"].map(scale)
    g = df.groupby("series_id")["y"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["roll_mean_13"] = g.transform(lambda s: s.shift(1).rolling(13).mean())
    df["roll_std_13"] = g.transform(lambda s: s.shift(1).rolling(13).std())
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["category_code"] = df["category"].astype("category").cat.codes
    return df


FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    "roll_mean_4",
    "roll_mean_13",
    "roll_std_13",
    "week_of_year",
    "category_code",
]


def global_gbm(train: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """One gradient-boosting model for all series, forecasting recursively week by week.

    Poisson loss models the expected demand directly, which suits counts with many zeros and
    avoids the under-forecasting you get from back-transforming a log-target model.
    """
    scale = train.groupby("series_id")["demand"].mean().clip(lower=1)
    feats = _features(train, scale).dropna(subset=["lag_52"])
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0
    )
    model.fit(feats[FEATURES], feats["y"])

    panel = train.copy()
    last_week = panel["week"].max()
    meta = panel.groupby("series_id")[["sku", "warehouse", "category"]].first()
    for step in range(1, horizon + 1):
        week = last_week + pd.Timedelta(weeks=step)
        future = meta.reset_index().assign(week=week, demand=np.nan)
        panel = pd.concat([panel, future], ignore_index=True)
        f = _features(panel, scale)
        rows = f["week"] == week
        pred = model.predict(f.loc[rows, FEATURES]) * f.loc[rows, "series_id"].map(scale).to_numpy()
        panel = panel.set_index(["series_id", "week"])
        panel.loc[
            list(zip(f.loc[rows, "series_id"], f.loc[rows, "week"], strict=True)), "demand"
        ] = pred
        panel = panel.reset_index()
    out = panel[panel["week"] > last_week].sort_values(["series_id", "week"])
    return {sid: g["demand"].to_numpy() for sid, g in out.groupby("series_id")}


LOCAL_MODELS = {
    "Seasonal naive": seasonal_naive,
    "Moving average (8w)": moving_average,
    "Holt-Winters": holt_winters,
}
