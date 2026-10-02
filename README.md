# Demand Forecasting & Inventory Planning

![CI](https://github.com/hellozaynabz-art/ds-supply-chain-forecasting/actions/workflows/ci.yml/badge.svg)

Better forecasts are only worth something if they change what sits on the shelf. This project
backtests four forecasting approaches across 200 SKU-warehouse series, then feeds each one into an
inventory simulation, so models are compared on **fill rate and stock held**, not just error metrics.

| Service vs. inventory | Example forecasts |
|---|---|
| ![tradeoff](reports/tradeoff.png) | ![examples](reports/examples.png) |

## What it shows

- **Honest backtesting**: forecasts made 13 weeks ahead using only earlier data; a separate earlier
  origin calibrates each model's error so safety stock never sees the evaluation period
- **Model range**: seasonal naive and moving-average baselines, Holt-Winters exponential smoothing
  per series, and one global gradient-boosting model with lag features and Poisson loss
  across all series ([`models.py`](src/demand_forecast/models.py))
- **Planning metrics**: WAPE, bias and MASE, the measures demand planners actually use
- **Inventory simulation**: weekly order-up-to policy with lead time, lost sales and safety stock
  sized for a 95% service target ([`backtest.py`](src/demand_forecast/backtest.py))
- **Real data quirks** handled in the Kaggle loader: returns stored as `(100)`, gaps filled with
  zero-demand weeks, and slow movers with intermittent demand

**Results on the Kaggle data (200 series, 2012–2016):** the 8-week moving average is both the most accurate
(WAPE 48.7%) and meets the 95% service target with **40% less inventory** than the most stock-heavy
model. On lumpy B2B demand, extra model complexity mostly added variance: an honest finding worth
having before investing in ML forecasting.

Full results: [`reports/REPORT.md`](reports/REPORT.md)

## Data

| Source | Contents | Command |
|---|---|---|
| Synthetic (default) | 200 series × 3 years with trend, yearly seasonality, promotions and slow movers | `make data` |
| Kaggle | [Forecasts for Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting): 1M+ orders from 4 warehouses; the 200 highest-volume series are kept | `make data SOURCE=kaggle` |

Kaggle needs a one-time sign-in: run `uvx kaggle auth login` (opens your browser).

## Run it

```bash
make setup      # install dependencies (requires uv)
make data       # build the weekly demand panel
make backtest   # forecast, score, simulate inventory; writes reports/
make test       # run the test suite
```

## Project structure

```
src/demand_forecast/
  data.py        synthetic generator + Kaggle loader → weekly demand panel
  models.py      baselines, Holt-Winters, global gradient boosting
  backtest.py    two-origin backtest, accuracy metrics, inventory simulation, report
tests/           model, metric and simulation tests
```

**Stack:** Python · pandas · statsmodels · scikit-learn · SciPy · Matplotlib · pytest · GitHub Actions
