import numpy as np

from demand_forecast import backtest, data, models


def test_synthetic_panel_is_complete():
    df = data.synthetic(n_series=5, weeks=60)
    assert df.groupby("series_id").size().eq(60).all()
    assert (df["demand"] >= 0).all()


def test_seasonal_naive_repeats_last_year():
    h = np.arange(104)
    assert (models.seasonal_naive(h, 4) == h[52:56]).all()


def test_holt_winters_tracks_a_trend():
    h = np.arange(120, dtype=float) + 50
    fc = models.holt_winters(h, 4)
    assert fc[0] > h[-5]


def test_global_gbm_returns_every_series():
    panel = data.synthetic(n_series=6, weeks=80)
    fc = models.global_gbm(panel, 3)
    assert set(fc) == set(panel["series_id"])
    assert all(len(v) == 3 and (v >= 0).all() for v in fc.values())


def test_perfect_forecast_has_zero_wape():
    act = {"a": np.array([10.0, 20.0])}
    acc = backtest.accuracy(act, act, {"a": np.arange(60.0)})
    assert acc["WAPE"] == 0 and acc["Bias"] == 0


def test_inventory_sim_more_safety_stock_means_higher_fill():
    rng = np.random.default_rng(0)
    demand = rng.poisson(20, 13).astype(float)
    fc = np.full(13, 20.0)
    low, _ = backtest.simulate_inventory(demand, fc, 0)
    high, stock = backtest.simulate_inventory(demand, fc, 30)
    assert high >= low and high > 0.95 and stock > 0
