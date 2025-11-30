# Quantitative Portfolio Risk Modeling – VaR-Based Risk Management System

This project implements an **end-to-end quantitative risk management pipeline** for a
6-stock global equity portfolio (US + India), combining:

- Classical **Value at Risk (VaR)** methods  
- **Backtesting** using Kupiec & Christoffersen tests  
- **Mean–VaR portfolio optimization**  
- **Machine learning models** for next-day return prediction  
- **Visualization dashboards** for risk and performance

The goal is to move beyond theoretical formulas and build a **practical, modular toolkit**
that can be extended for real-world portfolio risk analysis.

---

## 1. Universe & Data

**Assets:**
- US (NASDAQ): `AAPL`, `GOOGL`, `TSLA`
- India (NSE): `RELIANCE.NS`, `TCS.NS`, `INFY.NS`

**Data:**
- Source: `yfinance` (Yahoo Finance)
- Horizon: Last ~5 years of daily data
- FX: Daily USD/INR (`INR=X`) used to convert Indian equities to USD

Data is stored in:

- `data/combined_6stocks_5years.csv` – cleaned OHLCV for all 6 stocks in USD
- `data/portfolio_returns.csv` – daily portfolio returns and cumulative value

---

## 2. Project Structure

```text
data/                 # all CSV outputs
plots/                # generated plots
src/
  data_collection.py  # download prices, FX, build combined dataset
  portfolio_returns.py# compute log returns & equal-weight portfolio stats
  var_engine.py       # Historical, Parametric (Normal, t), Monte Carlo VaR
  backtesting.py      # Kupiec POF & Christoffersen conditional coverage
  optimization.py     # Mean–VaR optimization (VaR-constrained max return)
  ml_models.py        # ML models for next-day portfolio returns
  visualizations.py   # portfolio, VaR, exception, weights plots

Prerequisites

Python 3.10+ and:

pip install -r requirements.txt


Typical requirements:

yfinance
pandas
numpy
matplotlib
scipy
scikit-learn
xgboost   # optional, used if installed

Execution Order

From project root:

python src/data_collection.py
python src/portfolio_returns.py
python src/var_engine.py
python src/backtesting.py
python src/optimization.py
python src/ml_models.py
python src/visualizations.py


Outputs:

CSVs in data/

Plots in plots/

Console logs with metrics and statistics


Prerequisites

Python 3.10+ and:

pip install -r requirements.txt


Typical requirements:

yfinance
pandas
numpy
matplotlib
scipy
scikit-learn
xgboost   # optional, used if installed

Execution Order

From project root:

python src/data_collection.py
python src/portfolio_returns.py
python src/var_engine.py
python src/backtesting.py
python src/optimization.py
python src/ml_models.py
python src/visualizations.py


Outputs:

CSVs in data/

Plots in plots/

Console logs with metrics and statistics
