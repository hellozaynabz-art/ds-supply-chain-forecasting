.PHONY: setup data test lint all clean

setup:  ## install dependencies into .venv
	uv sync

data:  ## build the dataset (SOURCE=synthetic|kaggle)
	uv run python -m demand_forecast.data --source $(or $(SOURCE),synthetic)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

all: setup data test backtest

clean:
	rm -rf data/raw data/processed .pytest_cache .ruff_cache

backtest:  ## backtest models and simulate inventory; writes reports/REPORT.md
	uv run python -m demand_forecast.backtest
